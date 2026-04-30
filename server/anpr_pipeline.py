"""
ANPR Pipeline for Indian License Plates
Dual OCR: PaddleOCR (primary) + EasyOCR (fallback)

This is the ORIGINAL working pipeline, untouched except for:
  1. Added process_array() so the API can pass numpy arrays directly
  2. Added 'detections' to the return dict so the client can draw bboxes
  3. Made debug saving optional via save_debug flag
"""

import cv2
import numpy as np
import re
import os, sys, time, json
from pathlib import Path
from collections import Counter
from ultralytics import YOLO
import easyocr
import logging

try:
    import requests
except ImportError:
    requests = None

try:
    from paddleocr import PaddleOCR
    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

logging.basicConfig(level=logging.INFO, format='%(message)s')

def emit(d):
    print(json.dumps(d, ensure_ascii=False))

def _fetch_from_github():
    url = "https://raw.githubusercontent.com/datameet/india-rto-codes/master/rto_codes.json"
    resp = requests.get(url, timeout=10); resp.raise_for_status(); data = resp.json()
    codes = set()
    if isinstance(data, list):
        for e in data:
            if isinstance(e, dict) and 'state_code' in e: codes.add(e['state_code'].upper())
    elif isinstance(data, dict): codes = {k.upper() for k in data.keys()}
    return codes, url

def _fetch_from_wikipedia():
    url = "https://en.wikipedia.org/wiki/List_of_Regional_Transport_Office_districts_in_India"
    resp = requests.get(url, timeout=15, headers={'User-Agent': 'ANPR/1.0'}); resp.raise_for_status()
    matches = re.findall(r'\b([A-Z]{2})[-\s]?\d{1,2}\b', resp.text)
    counts = Counter(matches)
    bad = {'TD','TH','PX','EM','EN','IS','OF','TO','AT','BY','ON','IT','IF','OR','AN','NO','ID','IN'}
    return {c for c, n in counts.items() if n >= 2 and c not in bad}, url

def load_state_codes(cache_file='state_codes.json', ttl_days=30):
    cache = Path(cache_file)
    if cache.exists():
        try:
            age = (time.time() - cache.stat().st_mtime) / 86400
            if age < ttl_days:
                with open(cache) as f: codes = set(json.load(f))
                if len(codes) >= 30:
                    emit({"event":"state_codes","source":"cache","count":len(codes)}); return codes
        except: pass
    if requests is None: raise RuntimeError("No requests lib and no cache")
    for name, fn in [("github", _fetch_from_github), ("wikipedia", _fetch_from_wikipedia)]:
        try:
            emit({"event":"state_codes","source":name,"status":"fetching"})
            codes, url = fn()
            if len(codes) < 30: raise ValueError(f"Only {len(codes)} codes")
            with open(cache, 'w') as f: json.dump(sorted(codes), f)
            emit({"event":"state_codes","source":name,"status":"ok","count":len(codes)}); return codes
        except Exception as e:
            emit({"event":"warning","message":f"{name} failed: {e}"})
    if cache.exists():
        try:
            with open(cache) as f: codes = set(json.load(f))
            if len(codes) >= 30:
                emit({"event":"state_codes","source":"stale_cache"}); return codes
        except: pass
    raise RuntimeError("Cannot load state codes from any source")


class IndianANPR:
    def __init__(self, yolo_model_path='yolov8n.pt', debug_dir='debug_output',
                 blur_threshold=100.0, vote_frames=7, state_codes_cache='state_codes.json',
                 state_codes_ttl_days=30, clahe_weight=0.6, ocr_engine='auto',
                 save_debug=True):
        emit({"event":"init","status":"starting"})
        self.max_retries = 5
        self.save_debug = save_debug
        self.detector = YOLO(yolo_model_path)
        self.paddle_reader = None; self.easy_reader = None

        if ocr_engine in ('auto', 'both'):
            if HAS_PADDLE:
                self.paddle_reader = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=False)
            self.easy_reader = easyocr.Reader(['en'], gpu=False)
        elif ocr_engine == 'paddle':
            if not HAS_PADDLE: raise RuntimeError("PaddleOCR not installed")
            self.paddle_reader = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=False)
        elif ocr_engine == 'easyocr':
            self.easy_reader = easyocr.Reader(['en'], gpu=False)
        if not self.paddle_reader and not self.easy_reader: raise RuntimeError("No OCR engine")

        self.debug_dir = Path(debug_dir); self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.blur_threshold = blur_threshold; self.vote_frames = vote_frames
        self.clahe_weight = clahe_weight
        self.patterns = {
            "private": re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$"),
            "bh": re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{2}$"),
            "gov": re.compile(r"^[A-Z]{2}[0-9]{1,2}G[A-Z]{0,2}[0-9]{4}$"),
            "army": re.compile(r"^[0-9]{2}[A-Z][0-9]{5,6}[A-Z]$"),
        }
        self.letter_to_digit = {'O':'0','Q':'0','D':'0','I':'1','L':'1','Z':'2','S':'5','B':'8','G':'6'}
        self.digit_to_letter = {'0':'O','1':'I','2':'Z','5':'S','6':'G','8':'B'}
        self.state_codes = load_state_codes(state_codes_cache, state_codes_ttl_days)
        self._next_track_id = 1
        emit({"event":"init","status":"done","states":len(self.state_codes),
              "paddle":self.paddle_reader is not None,"easyocr":self.easy_reader is not None})

    def _tid(self):
        t = self._next_track_id; self._next_track_id += 1; return t

    def _save(self, path, img):
        if self.save_debug:
            cv2.imwrite(str(path), img)

    def detect_plate(self, image):
        results = self.detector(image, verbose=True)
        dets = []
        for r in results:
            if r.boxes is None: continue
            for box in r.boxes:
                c = float(box.conf[0])
                if c < 0.25: continue
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                dets.append(((x1,y1,x2,y2), c))
        dets.sort(key=lambda d: d[1], reverse=True); return dets

    def crop_plate(self, image, bbox, pad=5):
        h, w = image.shape[:2]; x1,y1,x2,y2 = bbox
        x1,y1 = max(0,x1-pad), max(0,y1-pad); x2,y2 = min(w,x2+pad), min(h,y2+pad)
        crop = image[y1:y2, x1:x2]
        ok = crop.size > 0 and crop.shape[0] > 5 and crop.shape[1] > 5
        emit({"event":"crop","bbox":list(bbox),"size":[crop.shape[1],crop.shape[0]],"ok":ok})
        return crop, ok

    def check_blur(self, gray):
        s = float(cv2.Laplacian(gray, cv2.CV_64F).var()); b = s < self.blur_threshold
        emit({"event":"blur","sharpness":round(s,2),"blurry":b}); return s, b

    def _brightness(self, gray):
        m = float(np.mean(gray))
        if m < 110:
            a = min(130.0/max(m,1.0), 2.5)
            return cv2.convertScaleAbs(gray, alpha=a, beta=10)
        return gray

    def _minimal(self, crop):
        img = crop.copy(); h,w = img.shape[:2]
        if h < 100: s=120/h; img=cv2.resize(img,(int(w*s),int(h*s)),interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
        return self._brightness(gray)

    def enhance_plate(self, plate_img):
        img = plate_img.copy(); h,w = img.shape[:2]
        if h < 100: s=120/h; img=cv2.resize(img,(int(w*s),int(h*s)),interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
        gray = self._brightness(gray)
        dn = cv2.fastNlMeansDenoising(gray, h=7)
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(dn)
        std = float(np.std(cl))
        if std < 50:
            he = cv2.equalizeHist(cl)
            cl = cv2.addWeighted(cl, self.clahe_weight, he, 1.0-self.clahe_weight, 0)
        cl = self._deskew(cl)
        bl = cv2.GaussianBlur(cl, (0,0), 2)
        return cv2.addWeighted(cl, 1.3, bl, -0.3, 0)

    def _variant(self, crop, v):
        img = crop.copy()
        if len(img.shape)==2: img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        h,w=img.shape[:2]
        if h<100: s=120/h; img=cv2.resize(img,(int(w*s),int(h*s)),interpolation=cv2.INTER_CUBIC)
        gray = self._brightness(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if v==1: _,out=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU); return out
        elif v==2:
            dn=cv2.fastNlMeansDenoising(gray,h=15)
            return cv2.adaptiveThreshold(dn,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,8)
        elif v==3:
            out=cv2.equalizeHist(gray); bl=cv2.GaussianBlur(out,(0,0),2)
            return cv2.addWeighted(out,1.8,bl,-0.8,0)
        elif v==4:
            dn=cv2.fastNlMeansDenoising(gray,h=20)
            k=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
            return cv2.createCLAHE(clipLimit=4.0,tileGridSize=(4,4)).apply(
                cv2.morphologyEx(dn,cv2.MORPH_OPEN,k))
        else:
            return cv2.createCLAHE(clipLimit=5.0,tileGridSize=(4,4)).apply(
                cv2.bilateralFilter(gray,9,75,75))

    def _deskew(self, gray):
        try:
            _,thr=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            coords=np.column_stack(np.where(thr>0))
            if len(coords)<20: return gray
            angle=cv2.minAreaRect(coords)[-1]
            angle = -(90+angle) if angle<-45 else -angle
            if abs(angle)<1 or abs(angle)>30: return gray
            h,w=gray.shape; M=cv2.getRotationMatrix2D((w//2,h//2),angle,1.0)
            return cv2.warpAffine(gray,M,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
        except: return gray

    def _geo_filter(self, dets, ih, iw):
        out = []
        for bbox, text, conf in dets:
            pts=np.array(bbox); ymin,ymax=pts[:,1].min(),pts[:,1].max()
            xmin=pts[:,0].min(); bh=ymax-ymin; cy=(ymin+ymax)/2
            if cy<ih*0.2 or cy>ih*0.8: continue
            if bh<ih*0.15: continue
            clean=re.sub(r'[^A-Z0-9]','',text.upper())
            if len(clean)<2: continue
            out.append((bbox,clean,conf,float(xmin)))
        return out

    def _assemble(self, filtered):
        filtered.sort(key=lambda r:r[3])
        return ''.join(r[1] for r in filtered), round(float(np.mean([r[2] for r in filtered])),3)

    def run_ocr(self, img, track_id, loose=False):
        ih, iw = img.shape[:2]; cands = []

        if self.paddle_reader:
            dets = []
            paddle_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
            #result = self.paddle_reader.ocr(paddle_img)
            result = self.paddle_reader.ocr(img , cls=True)  # --- IGNORE: old code without cls ---
            if result and result[0]:
                for line in result[0]:
                    dets.append((line[0], line[1][0], float(line[1][1])))
            if dets:
                f = self._geo_filter(dets, ih, iw)
                if f:
                    t,c = self._assemble(f); cands.append((t,c,'paddle'))
                    emit({"event":"ocr_paddle","tid":track_id,"text":t,"conf":c})
                else:
                    fb = [(b,re.sub(r'[^A-Z0-9]','',t.upper()),c,float(np.array(b)[:,0].min()))
                          for b,t,c in dets if len(re.sub(r'[^A-Z0-9]','',t.upper()))>=2]
                    if fb: t,c=self._assemble(fb); cands.append((t,c,'paddle_uf'))

        if self.easy_reader:
            tt,lt,ct = (0.3,0.2,0.05) if loose else (0.5,0.3,0.1)
            dets = self.easy_reader.readtext(img, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                detail=1,paragraph=False,text_threshold=tt,low_text=lt,
                contrast_ths=ct,adjust_contrast=0.7,mag_ratio=2.0)
            if dets:
                f = self._geo_filter(dets, ih, iw)
                if f:
                    t,c = self._assemble(f); cands.append((t,c,'easyocr'))
                    emit({"event":"ocr_easy","tid":track_id,"text":t,"conf":c})
                else:
                    fb = [(b,re.sub(r'[^A-Z0-9]','',t.upper()),c,float(np.array(b)[:,0].min()))
                          for b,t,c in dets if len(re.sub(r'[^A-Z0-9]','',t.upper()))>=2]
                    if fb: t,c=self._assemble(fb); cands.append((t,c,'easy_uf'))

        if not cands:
            emit({"event":"ocr","tid":track_id,"text":"","conf":0.0}); return "",0.0

        best = None
        for cand in cands:
            corrected = self.correct_text(re.sub(r'[^A-Z0-9]','',cand[0].upper()))
            if self._is_valid(corrected): best=cand; break
        if not best: best = max(cands, key=lambda c:c[1])

        emit({"event":"ocr","tid":track_id,"text":best[0],"conf":best[1],
              "engine":best[2],"candidates":len(cands)})
        return best[0], best[1]

    def _trim(self, text):
        cands=[text]; n=len(text)
        if n==11:
            for i in range(6,n): cands.append(text[:i]+text[i+1:])
            for i in range(4,6): cands.append(text[:i]+text[i+1:])
            for i in range(2,4): cands.append(text[:i]+text[i+1:])
        elif n==12:
            cands.append(text[:10])
            for i in range(6,10):
                for j in range(i+1,n): cands.append(text[:i]+text[i+1:j]+text[j+1:])
        return cands

    def correct_text(self, text):
        if not text: return text
        if self._is_valid(text): return text
        if len(text)<8 or len(text)>12: return text
        orig = text
        for cand in self._trim(text):
            n=len(cand)
            if n==10:
                c=self._fix10(cand)
                if self._is_valid(c):
                    emit({"event":"correct","in":orig,"out":c}); return c
                c=self._fixgov(cand)
                if self._is_valid(c):
                    emit({"event":"correct","in":orig,"out":c}); return c
            elif n==9:
                c=self._fix9(cand)
                if self._is_valid(c):
                    emit({"event":"correct","in":orig,"out":c}); return c
        return text

    def _fix10(self, t):
        c=list(t)
        for i in [0,1,4,5]:
            if c[i].isdigit(): c[i]=self.digit_to_letter.get(c[i],c[i])
        for i in [2,3,6,7,8,9]:
            if c[i].isalpha(): c[i]=self.letter_to_digit.get(c[i],c[i])
        return ''.join(c)

    def _fix9(self, t):
        c=list(t)
        for i in [0,1,3,4]:
            if c[i].isdigit(): c[i]=self.digit_to_letter.get(c[i],c[i])
        for i in [2,5,6,7,8]:
            if c[i].isalpha(): c[i]=self.letter_to_digit.get(c[i],c[i])
        return ''.join(c)

    def _fixgov(self, t):
        c=list(t)
        for i in [0,1]:
            if c[i].isdigit(): c[i]=self.digit_to_letter.get(c[i],c[i])
        for i in [2,3]:
            if c[i].isalpha(): c[i]=self.letter_to_digit.get(c[i],c[i])
        if c[4]=='6': c[4]='G'
        for i in [5,6]:
            if c[i].isdigit(): c[i]=self.digit_to_letter.get(c[i],c[i])
        for i in [7,8,9]:
            if c[i].isalpha(): c[i]=self.letter_to_digit.get(c[i],c[i])
        return ''.join(c)

    def normalize_plate(self, text):
        n=re.sub(r'[^A-Z0-9]','',text.upper().strip())
        emit({"event":"normalize","plate":n,"len":len(n)}); return n

    def _is_valid(self, text):
        if not text: return False
        mt=None
        for pt, pat in self.patterns.items():
            if pat.match(text): mt=pt; break
        if not mt: return False
        if mt in ('private','gov') and text[:2] not in self.state_codes: return False
        return True

    def validate_plate(self, text):
        v=self._is_valid(text); emit({"event":"validate","plate":text,"valid":v}); return v

    def _votes(self, readings, tid):
        if not readings: return "",0.0,0.0
        plates=[r[0] for r in readings if r[0]]; confs=[r[1] for r in readings if r[0]]
        emit({"event":"votes","tid":tid,"n":len(plates),"unique":len(set(plates)),"all":list(set(plates))})
        if not plates: return "",0.0,0.0
        ctr=Counter(plates); best,cnt=ctr.most_common(1)[0]
        stab=round(cnt/len(plates),3)
        mc=[c for p,c in zip(plates,confs) if p==best]
        return best, round(float(np.mean(mc)),3), stab

    # ---- ORIGINAL process() — unchanged ------------------------------------
    def process(self, image_path):
        tid=self._tid()
        emit({"event":"track","tid":tid,"image":str(image_path)})
        image=cv2.imread(str(image_path))
        if image is None: raise FileNotFoundError(f"Cannot read: {image_path}")
        return self._run_pipeline(image, tid, stem=Path(image_path).stem)

    # ---- NEW: process_array() — same logic, no disk read -------------------
    def process_array(self, image, source="memory"):
        tid=self._tid()
        emit({"event":"track","tid":tid,"image":source})
        stem = Path(source).stem if source != "memory" else f"mem_{tid}"
        return self._run_pipeline(image, tid, stem)

    # ---- Shared pipeline logic (extracted from original process()) ----------
    def _run_pipeline(self, image, tid, stem):
        dets=self.detect_plate(image)
        if not dets:
            emit({"event":"error","tid":tid,"msg":"No plate"})
            return {'success':False,'text':'','confidence':0.0,'stability':0.0,
                    'valid_format':False,'detections':[]}

        readings=[]

        for idx,(bbox,dc) in enumerate(dets):
            crop,ok=self.crop_plate(image,bbox)
            if not ok: continue
            self._save(self.debug_dir/f"{stem}_crop_{idx}.jpg",crop)
            enhanced=self.enhance_plate(crop)
            self._save(self.debug_dir/f"{stem}_enh_{idx}.jpg",enhanced)
            self.check_blur(enhanced)

            found=False; raw=""; norm=""; oc=0.0

            for att in range(self.max_retries+2):
                if att==0: img2=self._minimal(crop)
                elif att==1: img2=enhanced
                else:
                    emit({"event":"retry","tid":tid,"att":att}); img2=self._variant(crop,att-1)
                self._save(self.debug_dir/f"{stem}_att_{idx}_{att}.jpg",img2)

                raw,oc=self.run_ocr(img2,tid,loose=(att>=3))
                if not raw: continue
                corrected=self.correct_text(raw)
                norm=self.normalize_plate(corrected)
                v=self.validate_plate(norm)
                if v:
                    found=True; readings.append((norm,oc))
                    emit({"event":"valid","tid":tid,"att":att,"plate":norm,"conf":oc}); break
                if oc>=0.2: readings.append((norm,oc))

            if not found and raw and not any(r[0]==norm for r in readings):
                readings.append((norm,oc))
            if len(readings)>=self.vote_frames: break

        best,conf,stab=self._votes(readings,tid)
        emit({"event":"final","tid":tid,"plate":best,"conf":conf,"stab":stab,"valid":self._is_valid(best)})

        if dets and best:
            x1,y1,x2,y2=dets[0][0]; ann=image.copy()
            cv2.rectangle(ann,(x1,y1),(x2,y2),(0,255,0),3)
            cv2.putText(ann,best,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
            self._save(self.debug_dir/f"{stem}_result.jpg",ann)

        # Build bbox list for API response
        bbox_list = [
            {"x1":b[0],"y1":b[1],"x2":b[2],"y2":b[3],"confidence":round(c,3)}
            for b,c in dets
        ]

        return {'success':bool(best),'text':best,'confidence':conf,
                'stability':stab,'valid_format':self._is_valid(best),
                'detections':bbox_list}
