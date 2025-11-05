import cv2
import numpy as np
import os
import time
from collections import deque
import argparse

class PureCV_ISLRecognizer:
    def __init__(self, template_path):
        # Convert to absolute path and normalize
        self.template_path = os.path.abspath(os.path.normpath(template_path))
        
        # Core system state
        self.current_word = []
        self.words_history = []
        self.last_capture_time = 0
        self.last_gesture_time = 0
        self.inactivity_timeout = 3.0
        self.min_contour_area = 800
        
        # Recognition thresholds
        self.similarity_threshold = 0.7
        self.stability_threshold = 0.8
        
        # Countdown system
        self.countdown_start_time = 0
        self.countdown_active = False
        self.countdown_target = 2.0
        self.countdown_gesture = None
        
        # Display and performance
        self.frame_count = 0
        self.gestures_captured = 0
        self.processing_times = deque(maxlen=30)
        self.show_skin_mask = True
        
        # ROI configuration
        self.roi_active = True
        self.roi_size = 300
        self.roi_position = None
        
        # Skin detection parameters (optimized for brown hands)
        self.skin_lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
        self.skin_upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
        
        # Template storage
        self.templates = {}
        self.template_features = {}
        
        # Load and process templates
        self.load_templates()

    def load_templates(self):
        """Load and preprocess all template images"""
        print("Loading and processing templates...")
        print(f"Looking in: {self.template_path}")
        
        if not os.path.exists(self.template_path):
            print(f"Error: Template path does not exist!")
            print(f"Checked: {self.template_path}")
            print("\nPlease ensure the path is correct or use:")
            print("  python mainc.py --template_path \"C:\\Users\\iitsi\\Academics\\Sem V\\Computer Vision\\proj_2\\CV Project\\CV Project\\templates\"")
            return
            
        template_files = [f for f in os.listdir(self.template_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not template_files:
            print("No template images found!")
            print(f"Directory contents: {os.listdir(self.template_path)}")
            return
        
        print(f"Found {len(template_files)} template files")
        successful_templates = 0
        
        for filename in template_files:
            letter = os.path.splitext(filename)[0].upper()
            img_path = os.path.join(self.template_path, filename)
            template = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if template is not None:
                # Consistent preprocessing
                template_processed = self.preprocess_template(template)
                self.templates[letter] = template_processed
                
                # Extract features
                features = self.extract_pure_features(template_processed)
                if features is not None:
                    self.template_features[letter] = features
                    successful_templates += 1
                    print(f"  ✓ Loaded: {letter}")
                else:
                    print(f"  ✗ Failed features: {letter}")
            else:
                print(f"✗ Failed to load: {filename}")
        
        print(f"Successfully loaded {successful_templates}/{len(template_files)} templates")
        print(f"Available gestures: {list(self.template_features.keys())}")

    def preprocess_template(self, image):
        """Consistent template preprocessing to 200x200 with clean binarization"""
        target_size = 200
        
        # Handle different input sizes (like SPACE.jpg at 271x271)
        h, w = image.shape
        if h != target_size or w != target_size:
            scale = min(target_size / w, target_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Center on canvas
            canvas = np.zeros((target_size, target_size), dtype=np.uint8)
            x_offset = (target_size - new_w) // 2
            y_offset = (target_size - new_h) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            image = canvas
        
        # Apply CLAHE for contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(image)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Ensure white hand on black background
        if np.mean(thresh) > 127:
            thresh = cv2.bitwise_not(thresh)
        
        # Clean with morphology
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return thresh

    def preprocess_live_frame(self, image):
        """Enhanced skin segmentation using YCrCb color space"""
        # Convert to YCrCb for robust skin detection
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Apply CLAHE to Y channel for lighting invariance
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        ycrcb[:,:,0] = clahe.apply(ycrcb[:,:,0])
        
        # Skin color segmentation
        skin_mask = cv2.inRange(ycrcb, self.skin_lower_ycrcb, self.skin_upper_ycrcb)
        
        # Morphological cleaning
        kernel_open = np.ones((3, 3), np.uint8)
        kernel_close = np.ones((7, 7), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_open)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Fill largest contour to remove holes
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(skin_mask, [largest_contour], -1, 255, -1)
        
        return skin_mask

    def extract_pure_features(self, binary_image):
        """
        Extract pure computer vision features:
        - 7 Hu Moments (shape descriptors)
        - 5 contour shape features  
        - 1 edge density feature
        Total: 13 features
        """
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        
        # Basic area filter
        if contour_area < self.min_contour_area:
            return None
        
        try:
            # Get convex hull for stable shape
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            
            # 1. Hu Moments (7 features) - rotation, scale, translation invariant
            moments = cv2.moments(hull)
            hu_moments = cv2.HuMoments(moments).flatten()
            
            # Log transform for better numerical behavior
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-8)
            
            # 2. Contour shape features (5 features)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(hull, True)
            x, y, w, h = cv2.boundingRect(hull)
            
            aspect_ratio = w / h if h > 0 else 0
            extent = area / (w * h) if w * h > 0 else 0
            solidity = area / hull_area if hull_area > 0 else 0
            
            # 3. Edge density feature (1 feature)
            edges = cv2.Canny(binary_image, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.size + 1e-8)
            
            # Combine all 13 features
            features = np.concatenate([
                hu_moments,                    # 7 features
                [area / 10000,                 # Normalized area
                 perimeter / 1000,             # Normalized perimeter  
                 aspect_ratio, extent, solidity, # Shape ratios
                 edge_density]                 # Edge density
            ])
            
            return features
            
        except Exception as e:
            return None

    def cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two feature vectors"""
        if vec1 is None or vec2 is None:
            return 0.0
            
        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def recognize_by_similarity(self, live_features):
        """Recognize gesture using pure cosine similarity with templates"""
        if live_features is None or not self.template_features:
            return [], 0.0
        
        similarities = []
        
        for letter, template_feat in self.template_features.items():
            similarity = self.cosine_similarity(live_features, template_feat)
            similarities.append((letter, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top matches and best similarity
        top_matches = similarities[:3]
        best_similarity = top_matches[0][1] if top_matches else 0.0
        
        return top_matches, best_similarity

    def update_countdown(self, current_gesture, current_similarity):
        """Manage the 2-second countdown for gesture stability"""
        current_time = time.time()
        
        # Reset if no valid gesture or low similarity
        if not current_gesture or current_similarity < self.similarity_threshold:
            self.countdown_active = False
            return False, 0.0
        
        if self.countdown_active and self.countdown_gesture == current_gesture:
            # Continue existing countdown
            elapsed = current_time - self.countdown_start_time
            progress = min(elapsed / self.countdown_target, 1.0)
            
            if elapsed >= self.countdown_target:
                # Countdown complete
                self.countdown_active = False
                return True, 1.0
            else:
                return False, progress
        else:
            # Start new countdown
            self.countdown_active = True
            self.countdown_gesture = current_gesture
            self.countdown_start_time = current_time
            return False, 0.0

    def commit_gesture(self, gesture):
        """Add gesture to current word"""
        current_time = time.time()
        self.last_gesture_time = current_time
        
        if gesture == "SPACE":
            if self.current_word:
                word = ''.join(self.current_word)
                self.words_history.append(word)
                self.current_word = []
                print(f"WORD COMPLETED: '{word}'")
        else:
            self.current_word.append(gesture)
            self.gestures_captured += 1
            print(f"✅ GESTURE CAPTURED: '{gesture}' -> Current: '{''.join(self.current_word)}'")
        
        # Reset countdown
        self.countdown_active = False
        self.last_capture_time = current_time

    def setup_roi(self, frame):
        """Setup ROI in center of frame"""
        h, w = frame.shape[:2]
        roi_x = (w - self.roi_size) // 2
        roi_y = (h - self.roi_size) // 2
        self.roi_position = (roi_x, roi_y, self.roi_size, self.roi_size)
        return self.roi_position

    def extract_roi(self, frame):
        """Extract ROI from frame"""
        if not self.roi_active or self.roi_position is None:
            return frame
        x, y, size, _ = self.roi_position
        roi = frame[y:y+size, x:x+size]
        # Handle case where ROI is out of bounds
        return roi if roi.size > 0 else frame

    def create_display(self, frame, skin_mask, contours, top_matches, countdown_progress, processing_time):
        """Create clean, informative display"""
        h, w = frame.shape[:2]
        
        # Create side-by-side display
        display_frame = np.zeros((h, w * 2, 3), dtype=np.uint8)
        
        # Left: Original frame with annotations
        left_frame = frame.copy()
        
        # Draw ROI with state-based coloring
        if self.roi_active and self.roi_position:
            x, y, size, _ = self.roi_position
            if countdown_progress >= 1.0:
                color = (0, 255, 0)  # Green - ready
            elif countdown_progress > 0:
                color = (0, 255, 255)  # Yellow - counting
            else:
                color = (128, 128, 128)  # Gray - idle
            
            cv2.rectangle(left_frame, (x, y), (x + size, y + size), color, 3)
            cv2.putText(left_frame, "ROI", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw contours if valid
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > self.min_contour_area:
                hull = cv2.convexHull(largest_contour)
                cv2.drawContours(left_frame, [hull], -1, (0, 255, 0), 2)
                cv2.drawContours(left_frame, [largest_contour], -1, (255, 0, 0), 1)
        
        # Right: Information and skin mask
        right_frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Top: Skin mask display
        skin_display = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR)
        skin_display = cv2.resize(skin_display, (w, h//3))
        right_frame[0:h//3, :] = skin_display
        cv2.putText(right_frame, "Hand Segmentation", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Middle: Recognition info
        info_panel = np.zeros((h//3, w, 3), dtype=np.uint8)
        y_offset = 30
        
        # Current detection status
        if top_matches:
            best_gesture, best_similarity = top_matches[0]
            status_color = (0, 255, 0) if countdown_progress >= 1.0 else (0, 255, 255)
            status_text = f"READY: {best_gesture}" if countdown_progress >= 1.0 else f"Detecting: {best_gesture}"
            cv2.putText(info_panel, status_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            y_offset += 35
            
            cv2.putText(info_panel, f"Similarity: {best_similarity:.1%}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        else:
            cv2.putText(info_panel, "No hand detected", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 35
        
        # Countdown progress bar
        bar_width = 200
        bar_height = 20
        cv2.rectangle(info_panel, (10, y_offset), (10 + bar_width, y_offset + bar_height), 
                     (50, 50, 50), -1)
        
        if countdown_progress > 0:
            progress_width = int(bar_width * countdown_progress)
            color = (0, 255, 0) if countdown_progress >= 1.0 else (0, 255, 255)
            cv2.rectangle(info_panel, (10, y_offset), (10 + progress_width, y_offset + bar_height), 
                         color, -1)
        
        countdown_text = "READY!" if countdown_progress >= 1.0 else f"Stability: {countdown_progress:.0%}"
        cv2.putText(info_panel, countdown_text, (10 + bar_width + 10, y_offset + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 30
        
        right_frame[h//3:2*h//3, :] = info_panel
        
        # Bottom: Top matches
        match_panel = np.zeros((h//3, w, 3), dtype=np.uint8)
        y_offset = 25
        
        cv2.putText(match_panel, "TOP MATCHES:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        
        for i, (letter, similarity) in enumerate(top_matches[:3]):
            color = (0, 255, 0) if similarity > 0.8 else (0, 255, 255) if similarity > 0.6 else (255, 255, 0)
            match_text = f"{letter}: {similarity:.1%}"
            cv2.putText(match_panel, match_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 22
        
        right_frame[2*h//3:, :] = match_panel
        
        # Combine left and right
        display_frame[:, :w] = left_frame
        display_frame[:, w:] = right_frame
        
        # System status overlay
        current_word_text = ''.join(self.current_word) if self.current_word else "[Ready...]"
        history_text = " | ".join(self.words_history[-3:]) if self.words_history else "No words yet"
        
        cv2.putText(display_frame, f"WORD: {current_word_text}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, f"History: {history_text}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Performance metrics
        metrics_y = h - 10
        cv2.putText(display_frame, f"FPS: {self.calculate_fps():.1f}", 
                   (w - 150, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Gestures: {self.gestures_captured}", 
                   (w - 300, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Process: {processing_time:.1f}ms", 
                   (w - 450, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return display_frame

    def calculate_fps(self):
        """Calculate and return current FPS"""
        if not hasattr(self, 'fps_start_time'):
            self.fps_start_time = time.time()
            self.fps_frame_count = 0
            return 0
        
        self.fps_frame_count += 1
        elapsed = time.time() - self.fps_start_time
        
        if elapsed >= 1.0:
            self.current_fps = self.fps_frame_count / elapsed
            self.fps_start_time = time.time()
            self.fps_frame_count = 0
        
        return getattr(self, 'current_fps', 0)

    def process_frame(self, frame):
        """Process single frame with pure CV pipeline"""
        start_time = time.time()
        self.frame_count += 1
        current_time = time.time()
        
        # Setup ROI on first frame
        if self.roi_position is None:
            self.setup_roi(frame)
        
        # Extract and process ROI
        roi_frame = self.extract_roi(frame)
        skin_mask = self.preprocess_live_frame(roi_frame)
        
        # Extract features
        features = self.extract_pure_features(skin_mask)
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Recognize gesture
        top_matches, best_similarity = self.recognize_by_similarity(features)
        best_gesture = top_matches[0][0] if top_matches else None
        
        # Update countdown system
        countdown_complete, countdown_progress = self.update_countdown(best_gesture, best_similarity)
        
        # Update timing for valid detections
        if best_gesture and best_similarity > 0.5:
            self.last_gesture_time = current_time
        
        # Commit gesture if countdown completes
        if countdown_complete and best_gesture and current_time - self.last_capture_time > 1.0:
            self.commit_gesture(best_gesture)
        
        # Auto-complete word on inactivity
        if self.should_finalize_word() and self.current_word:
            word = ''.join(self.current_word)
            self.words_history.append(word)
            self.current_word = []
            print(f"⏱ AUTO-COMPLETE: '{word}'")
            self.last_gesture_time = current_time
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        self.processing_times.append(processing_time)
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        # Create display
        display_frame = self.create_display(
            frame, skin_mask, contours, top_matches, countdown_progress, avg_processing_time
        )
        
        return display_frame

    def should_finalize_word(self):
        """Check if word should be finalized due to inactivity"""
        if not self.current_word:
            return False
        return time.time() - self.last_gesture_time > self.inactivity_timeout

    def run(self):
        """Main execution loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        # Configure camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("🎥 Camera initialized successfully!")
        print(f"📚 Loaded {len(self.template_features)} gestures: {list(self.template_features.keys())}")
        print("🔧 PIPELINE:")
        print("   ➤ YCrCb skin segmentation + CLAHE")
        print("   ➤ 13 geometric features (Hu Moments + shape + edges)")
        print("   ➤ Cosine similarity template matching")
        print("   ➤ 2-second stability countdown")
        print("\n🎮 CONTROLS:")
        print("   SPACE - Insert space/complete word")
        print("   'c'   - Clear current word")
        print("   'm'   - Toggle skin mask display")
        print("   'q'   - Quit application")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame")
                    break
                
                # Mirror frame for more intuitive interaction
                frame = cv2.flip(frame, 1)
                
                # Process frame through pure CV pipeline
                display_frame = self.process_frame(frame)
                
                # Display result
                cv2.imshow('ISL Recognition - Pure Computer Vision', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.current_word = []
                    self.countdown_active = False
                    print("🗑 Current word cleared")
                elif key == ord('m'):
                    self.show_skin_mask = not self.show_skin_mask
                    print(f"👁 Skin mask: {'ON' if self.show_skin_mask else 'OFF'}")
                elif key == ord(' '):
                    if self.current_word:
                        word = ''.join(self.current_word)
                        self.words_history.append(word)
                        self.current_word = []
                        print(f"⌨ MANUAL SPACE - Word: '{word}'")
                
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        # Session summary
        print("\n📊 SESSION SUMMARY:")
        print(f"   Frames processed: {self.frame_count}")
        print(f"   Gestures captured: {self.gestures_captured}")
        print(f"   Words recognized: {self.words_history}")
        if self.processing_times:
            avg_time = np.mean(self.processing_times)
            print(f"   Avg processing time: {avg_time:.1f}ms")
            print(f"   Theoretical max FPS: {1000/avg_time:.1f}")

def main():
    parser = argparse.ArgumentParser(description='Pure Computer Vision ISL Recognition')
    parser.add_argument('--template_path', type=str, 
                       default=r'C:\Users\iitsi\Academics\Sem V\Computer Vision\proj_2\CV Project\CV Project\templates',
                       help='Path to template images directory')
    
    args = parser.parse_args()
    
    # Initialize and run the pure CV recognizer
    recognizer = PureCV_ISLRecognizer(args.template_path)
    recognizer.run()

if __name__ == "__main__":
    main()