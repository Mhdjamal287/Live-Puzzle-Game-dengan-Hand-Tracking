import cv2
import mediapipe as mp
import numpy as np
import random
import time

class UltimateAestheticPuzzle:
    def __init__(self):
        # Inisialisasi Kamera HD
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Inisialisasi MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2, 
            min_detection_confidence=0.8, 
            min_tracking_confidence=0.8
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Style Iron Man (Minimalist White)
        # Menghilangkan warna merah default, diganti putih bersih
        self.hand_conn_style = self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1)
        self.hand_joint_style = self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
        
        # State Aplikasi
        self.mode = "setup" 
        self.grid_size = 3
        self.tiles = []  
        self.tile_rects = [] 
        self.tile_order = [] 
        
        self.selected_tile_idx = None
        self.last_cursor_pos = (0, 0)
        self.reset_cooldown = 0
        self.start_time = 0
        self.elapsed_time = 0
        self.is_solved = False

    def get_dist(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def draw_rounded_rect(self, img, pt1, pt2, color, thickness, radius=25):
        x1, y1 = pt1
        x2, y2 = pt2
        if thickness == -1:
            cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
            cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
            cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
            cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
            cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
        else:
            cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
            cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
            cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
            cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
            cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
            cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

    def is_metal_gesture(self, lms):
        """Deteksi gesture Metal (Ibu Jari, Telunjuk, Kelingking UP | Tengah, Manis DOWN)"""
        # Landmark: Ibu jari(4), Telunjuk(8), Tengah(12), Manis(16), Kelingking(20)
        index_open = lms.landmark[8].y < lms.landmark[6].y
        middle_closed = lms.landmark[12].y > lms.landmark[10].y
        ring_closed = lms.landmark[16].y > lms.landmark[14].y
        pinky_open = lms.landmark[20].y < lms.landmark[18].y
        # Jempol opsional karena kadang sulit dideteksi stabil, fokus ke 4 jari utama
        return index_open and pinky_open and middle_closed and ring_closed

    def create_puzzle_snapshot(self, frame, roi):
        x1, y1, x2, y2 = roi
        img_crop = frame[y1:y2, x1:x2].copy()
        h_crop, w_crop, _ = img_crop.shape
        th, tw = h_crop // self.grid_size, w_crop // self.grid_size
        
        self.tiles = []
        self.tile_rects = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                tile = img_crop[i*th:(i+1)*th, j*tw:(j+1)*tw].copy()
                self.tiles.append(tile)
                # Slot grid tetap di layar
                self.tile_rects.append([j*tw + x1, i*th + y1, tw, th])
        
        self.tile_order = list(range(len(self.tiles)))
        random.shuffle(self.tile_order)
        self.start_time = time.time()
        self.is_solved = False

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            cursor = None
            is_pinching = False
            metal_detected = False

            # Deteksi Landmark Tangan
            if results.multi_hand_landmarks:
                for lms in results.multi_hand_landmarks:
                    if self.is_metal_gesture(lms): 
                        metal_detected = True
                    
                    idx_tip = (int(lms.landmark[8].x * w), int(lms.landmark[8].y * h))
                    thm_tip = (int(lms.landmark[4].x * w), int(lms.landmark[4].y * h))
                    
                    # Cek Pinch (Jarak Telunjuk ke Jempol)
                    if self.get_dist(idx_tip, thm_tip) < 45: 
                        is_pinching = True
                        cursor = idx_tip
                        self.last_cursor_pos = idx_tip

            # Logika Reset (Gesture Metal)
            if metal_detected and self.mode == "puzzle" and self.reset_cooldown == 0:
                self.mode = "setup"
                self.is_solved = False
                self.reset_cooldown = 45

            # --- MODE SETUP ---
            if self.mode == "setup":
                if results.multi_hand_landmarks:
                    for lms in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(frame, lms, self.mp_hands.HAND_CONNECTIONS, 
                                                    self.hand_joint_style, self.hand_conn_style)

                if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
                    p1 = (int(results.multi_hand_landmarks[0].landmark[8].x * w), int(results.multi_hand_landmarks[0].landmark[8].y * h))
                    p2 = (int(results.multi_hand_landmarks[1].landmark[8].x * w), int(results.multi_hand_landmarks[1].landmark[8].y * h))
                    cx, cy = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
                    side = max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]), 250)
                    x1, y1 = max(0, cx - side//2), max(0, cy - side//2)
                    x2, y2 = min(w, x1 + side), min(h, y1 + side)
                    self.draw_rounded_rect(frame, (x1, y1), (x2, y2), (255, 255, 255), 1, radius=20)
                    
                    if is_pinching:
                        self.create_puzzle_snapshot(frame, (x1, y1, x2, y2))
                        self.mode = "puzzle"

            # --- MODE PUZZLE ---
            elif self.mode == "puzzle":
                if not self.is_solved:
                    self.elapsed_time = time.time() - self.start_time
                    # Cek Menang
                    if all(self.tile_order[i] == i for i in range(len(self.tile_order))):
                        self.is_solved = True

                # 1. Gambar Puzzle
                for i in range(len(self.tile_order)):
                    tx, ty, tw, th = self.tile_rects[i]
                    if self.selected_tile_idx == i and cursor:
                        tx, ty = cursor[0] - tw//2, cursor[1] - th//2
                    
                    if 0 <= ty < h-th and 0 <= tx < w-tw:
                        frame[ty:ty+th, tx:tx+tw] = self.tiles[self.tile_order[i]]
                        self.draw_rounded_rect(frame, (tx, ty), (tx+tw, ty+th), (0, 0, 0), 2, radius=10)

                # 2. Logika Drag & Drop
                if is_pinching and cursor and not self.is_solved:
                    if self.selected_tile_idx is None:
                        for i, (rx, ry, rw, rh) in enumerate(self.tile_rects):
                            if rx < cursor[0] < rx+rw and ry < cursor[1] < ry+rh:
                                self.selected_tile_idx = i
                                break
                else:
                    if self.selected_tile_idx is not None:
                        lx, ly = self.last_cursor_pos
                        for i, (rx, ry, rw, rh) in enumerate(self.tile_rects):
                            if rx < lx < rx+rw and ry < ly < ry+rh:
                                self.tile_order[self.selected_tile_idx], self.tile_order[i] = \
                                self.tile_order[i], self.tile_order[self.selected_tile_idx]
                                break
                        self.selected_tile_idx = None

                # 3. Tampilan Clear
                if self.is_solved:
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0,0), (w, h), (0,0,0), -1)
                    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                    
                    msg = "SYSTEM STABILIZED"
                    (tw_m, th_m), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
                    cv2.putText(frame, msg, (w//2-tw_m//2, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                    
                    hint = "Show METAL gesture 🤟 to reboot"
                    (tw_h, th_h), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
                    cv2.putText(frame, hint, (w//2-tw_h//2, h//2+50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

                # 4. Gambar Hand Tracking (White Style)
                if results.multi_hand_landmarks:
                    for lms in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(frame, lms, self.mp_hands.HAND_CONNECTIONS, 
                                                    self.hand_joint_style, self.hand_conn_style)
                        
                        # Efek Glow saat Pinch
                        if is_pinching and cursor:
                            cv2.circle(frame, cursor, 15, (255, 255, 255), 1, cv2.LINE_AA)
                            cv2.circle(frame, cursor, 5, (255, 255, 255), -1)

                # 5. Aesthetic UI Info
                txt = f"ACTIVE_TIME: {int(self.elapsed_time)}s"
                cv2.putText(frame, txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            if self.reset_cooldown > 0: self.reset_cooldown -= 1
            cv2.imshow("STARK_INTERFACE_V3", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = UltimateAestheticPuzzle()
    app.run()