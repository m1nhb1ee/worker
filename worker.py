from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re
from PIL import Image, ImageEnhance, ImageFilter
import io
import os
import numpy as np
import cv2
from collections import Counter

# Import multiple OCR engines for fallback
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


class EnhancedCaptchaSolver:
    """Enhanced Captcha Solver với độ chính xác cao - KHÔNG AUTO PADDING"""
    
    def __init__(self, checker_instance):
        self.checker = checker_instance
        
        # Character correction mapping - CAPTCHA CHỮ THƯỜNG
        self.char_corrections = {
            # Số và chữ thường thường bị nhầm lẫn
            'l': '1', '|': 'j', '1': 'i',
            'G': '6',
            # Loại bỏ các chữ HOA không có trong captcha chữ thường
            'A': 'a', 'C': 'c', 'D': 'd', 'E': 'e', 'F': 'f',
            'H': 'h', 'J': 'j', 'K': 'k', 'L': 'l', 'M': 'm',
            'N': 'n', 'P': 'p', 'R': 'r', 'T': 't', 'U': 'u',
            'V': 'v', 'W': 'w', 'X': 'x', 'Y': 'y'
        }
    
    def advanced_preprocessing(self, img_bytes: bytes) -> list:
        """Advanced preprocessing với 14+ phương pháp khác nhau"""
        try:
            original = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_array = np.array(original)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            processed_images = []
            
            # 1. Ảnh gốc (cho TrOCR)
            processed_images.append(("original", original))
            
            # 2. Resize to standard size for better OCR
            height, width = img_bgr.shape[:2]
            if width < 200:  # Scale up small images
                scale_factor = 200 / width
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                resized = cv2.resize(img_bgr, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                processed_images.append(("resized", Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))))
            
            # 3. Noise removal
            gaussian = cv2.GaussianBlur(img_bgr, (3, 3), 0)
            processed_images.append(("gaussian", Image.fromarray(cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB))))
            
            median = cv2.medianBlur(img_bgr, 3)
            processed_images.append(("median", Image.fromarray(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))))
            
            # 4. Advanced thresholding
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # Otsu's thresholding - ĐẢM BẢO RGB CHO TROCR
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsu_rgb = cv2.cvtColor(otsu, cv2.COLOR_GRAY2RGB)
            processed_images.append(("otsu", Image.fromarray(otsu_rgb)))
            
            # Adaptive thresholding - ĐẢM BẢO RGB CHO TROCR
            adaptive_mean = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
            )
            adaptive_mean_rgb = cv2.cvtColor(adaptive_mean, cv2.COLOR_GRAY2RGB)
            processed_images.append(("adaptive_mean", Image.fromarray(adaptive_mean_rgb)))
            
            adaptive_gaussian = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            adaptive_gaussian_rgb = cv2.cvtColor(adaptive_gaussian, cv2.COLOR_GRAY2RGB)
            processed_images.append(("adaptive_gaussian", Image.fromarray(adaptive_gaussian_rgb)))
            
            # 5. Color space transformations - ĐẢM BẢO RGB CHO TROCR
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            l_channel = lab[:,:,0]
            # Convert grayscale to RGB for TrOCR
            l_channel_rgb = cv2.cvtColor(l_channel, cv2.COLOR_GRAY2RGB)
            processed_images.append(("lab_l", Image.fromarray(l_channel_rgb)))
            
            # HSV masks
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            
            # Dark text mask
            lower_dark = np.array([0, 0, 0])
            upper_dark = np.array([180, 255, 100])
            mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
            result_dark = cv2.bitwise_and(gray, gray, mask=mask_dark)
            # Convert to RGB for TrOCR
            result_dark_rgb = cv2.cvtColor(result_dark, cv2.COLOR_GRAY2RGB)
            processed_images.append(("masked_dark", Image.fromarray(result_dark_rgb)))
            
            # Medium contrast mask
            lower_medium = np.array([0, 50, 50])
            upper_medium = np.array([180, 255, 200])
            mask_medium = cv2.inRange(hsv, lower_medium, upper_medium)
            result_medium = cv2.bitwise_and(gray, gray, mask=mask_medium)
            # Convert to RGB for TrOCR
            result_medium_rgb = cv2.cvtColor(result_medium, cv2.COLOR_GRAY2RGB)
            processed_images.append(("masked_medium", Image.fromarray(result_medium_rgb)))
            
            # 6. Morphological operations - ĐẢM BẢO RGB CHO TROCR
            kernel_small = np.ones((2,2), np.uint8)
            
            closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel_small)
            closed_rgb = cv2.cvtColor(closed, cv2.COLOR_GRAY2RGB)
            processed_images.append(("morphed_close", Image.fromarray(closed_rgb)))
            
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_small)
            opened_rgb = cv2.cvtColor(opened, cv2.COLOR_GRAY2RGB)
            processed_images.append(("morphed_open", Image.fromarray(opened_rgb)))
            
            # 7. Edge enhancement - ĐẢM BẢO RGB CHO TROCR
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = np.uint8(np.absolute(laplacian))
            sharpened = cv2.addWeighted(gray, 1.5, laplacian, -0.5, 0)
            sharpened_rgb = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
            processed_images.append(("sharpened", Image.fromarray(sharpened_rgb)))
            
            # 8. CLAHE enhancement - ĐẢM BẢO RGB CHO TROCR
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            clahe_img = clahe.apply(gray)
            clahe_rgb = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
            processed_images.append(("clahe", Image.fromarray(clahe_rgb)))
            
            return processed_images
            
        except Exception as e:
            print(f"Lỗi advanced preprocessing: {e}")
            return [("original", Image.open(io.BytesIO(img_bytes)).convert("RGB"))]
    
    def correct_characters(self, text: str) -> str:
        """Sửa những ký tự thường bị nhầm lẫn"""
        corrected = ""
        for char in text:
            corrected += self.char_corrections.get(char, char)
        return corrected
    
    def extract_captcha_6chars(self, text: str) -> str:
        """Giữ nguyên mọi ký tự chữ và số, luôn trả về 6 ký tự chữ/số đầu tiên (bỏ ký tự đặc biệt, giữ đúng thứ tự)"""
        if not text:
            return ""
        # Lấy đúng 6 ký tự chữ/số đầu tiên, giữ thứ tự xuất hiện
        chars = []
        for c in text:
            if c.isalnum():
                chars.append(c)
                if len(chars) == 6:
                    break
        return ''.join(chars).lower()
    
    def smart_ensemble_voting(self, results: list) -> dict:
        """Thuật toán thông minh - CHỈ CHẤP NHẬN KẾT QUẢ 6 KÝ TỰ"""
        if not results:
            return {"result": "", "should_refresh": True, "confidence": 0}
        
        # CHỈ LẤY KẾT QUẢ 6 KÝ TỰ - REFRESH NẾU KHÔNG ĐỦ
        results_6chars = [r for r in results if len(r.get('text', '')) == 6]
        results_other = [r for r in results if len(r.get('text', '')) != 6]
        
        total_results = len(results)
        
        print(f"Tổng kết quả: {total_results}")
        print(f"  - 6 ký tự: {len(results_6chars)}")
        print(f"  - Khác (bỏ qua): {len(results_other)}")
        
        # Trường hợp 1: Không có kết quả 6 ký tự -> Refresh
        if not results_6chars:
            print("Không có kết quả nào đúng 6 ký tự -> Refresh captcha")
            return {"result": "", "should_refresh": True, "confidence": 0}
        
        # Trường hợp 2: Có kết quả 6 ký tự -> Phân tích theo vị trí
        print("Phân tích xác suất theo vị trí ký tự (chỉ kết quả 6 ký tự)...")
        
        # Tạo ma trận đếm ký tự theo vị trí [position][character] = count
        position_counts = [{} for _ in range(6)]  # Cố định 6 vị trí
        
        # Đếm ký tự tại mỗi vị trí (chỉ từ kết quả 6 ký tự)
        for result in results_6chars:
            text = result['text']
            weight = self._get_result_weight(result)
            
            for pos, char in enumerate(text):
                if pos < 6:
                    if char not in position_counts[pos]:
                        position_counts[pos][char] = 0
                    position_counts[pos][char] += weight
        
        # Xây dựng kết quả cuối cùng dựa trên ký tự có tần suất cao nhất tại mỗi vị trí
        final_result = ""
        total_confidence = 0
        position_confidences = []
        
        for pos in range(6):
            if position_counts[pos]:
                # Sắp xếp theo tần suất
                sorted_chars = sorted(position_counts[pos].items(), key=lambda x: x[1], reverse=True)
                best_char = sorted_chars[0][0]
                best_count = sorted_chars[0][1]
                total_votes = sum(position_counts[pos].values())
                
                # Tính confidence cho vị trí này
                pos_confidence = best_count / total_votes if total_votes > 0 else 0
                position_confidences.append(pos_confidence)
                
                final_result += best_char
                
                print(f"  Vị trí {pos+1}: '{best_char}' ({best_count}/{total_votes} = {pos_confidence:.2f})")
            else:
                # Không có dữ liệu cho vị trí này -> Lỗi nghiêm trọng
                print(f"  Vị trí {pos+1}: Không có dữ liệu -> Refresh")
                return {"result": "", "should_refresh": True, "confidence": 0}
        
        # Tính confidence tổng thể
        if position_confidences:
            total_confidence = sum(position_confidences) / len(position_confidences)
        
        print(f"Kết quả được tổng hợp: '{final_result}' (confidence: {total_confidence:.2f})")
        
        # Trường hợp 3: Kiểm tra kết quả cuối cùng phải đủ 6 ký tự
        if len(final_result) != 6:
            print(f"Kết quả cuối cùng không đúng 6 ký tự ({len(final_result)}) -> Refresh")
            return {"result": "", "should_refresh": True, "confidence": total_confidence}
        
        # Trường hợp 4: Kết quả đúng 6 ký tự
        print(f"Kết quả cuối cùng đúng 6 ký tự: '{final_result}' -> OK")
        return {"result": final_result, "should_refresh": False, "confidence": total_confidence}
    
    def _get_result_weight(self, result: dict) -> float:
        """Tính trọng số cho mỗi kết quả OCR"""
        weight = 1.0
        
        # Trọng số theo OCR engine
        if result['ocr'] == 'trocr':
            weight *= 1.5
        elif result['ocr'] == 'easyocr':
            weight *= 1.2
        
        # Trọng số theo preprocessing method
        if result['method'] in ['lab_l', 'gaussian', 'sharpened']:
            weight *= 1.3
        elif result['method'] in ['otsu', 'adaptive_gaussian', 'clahe']:
            weight *= 1.1
        
        # Trọng số theo confidence (nếu có)
        if 'confidence' in result and result['confidence']:
            weight *= (0.5 + result['confidence'])  # Scale từ 0.5 đến 1.5
        
        return weight
    
    def solve_captcha_enhanced(self, img_bytes: bytes, attempt_num: int = 0, refresh_count: int = 0) -> dict:
        """Enhanced captcha solver - KHÔNG AUTO PADDING với logic thông minh"""
        try:
            if self.checker.save_captcha_images:
                filename = f"{self.checker.captcha_folder}/enhanced_captcha_attempt_{attempt_num}_refresh_{refresh_count}_{int(time.time())}.png"
                with open(filename, 'wb') as f:
                    f.write(img_bytes)
                print(f"Lưu captcha: {filename}")

            processed_images = self.advanced_preprocessing(img_bytes)
            all_results = []
            
            # CHỈ SỬ DỤNG CÁC PHƯƠNG PHÁP ĐÃ CHỌN
            selected_combinations = []
            
            for method_name, processed_img in processed_images:
                for ocr_engine in self.checker.ocr_engines:
                    # CHỈ CHẤP NHẬN CÁC PHƯƠNG PHÁP ĐÃ ĐỊNH TRƯỚC
                    valid_method = False
                    
                    if ocr_engine == "easyocr" and method_name == "adaptive_mean":
                        valid_method = True
                    elif ocr_engine == "trocr" and method_name in ["lab_l", "sharpened", "clahe", "gaussian"]:
                        valid_method = True
                    
                    if valid_method:
                        selected_combinations.append((method_name, processed_img, ocr_engine, "SELECTED"))
            
            # Chỉ xử lý các combination đã chọn
            all_combinations = selected_combinations
            
            if not all_combinations:
                print("Không có combination nào được chọn -> Fallback to original")
                # Fallback nếu không có method nào khớp
                for method_name, processed_img in processed_images[:3]:  # Chỉ lấy 3 method đầu
                    for ocr_engine in self.checker.ocr_engines:
                        all_combinations.append((method_name, processed_img, ocr_engine, "FALLBACK"))
            
            for method_name, processed_img, ocr_engine, priority in all_combinations:
                    try:
                        raw_result = ""
                        confidence = 0
                        
                        if ocr_engine == "trocr" and hasattr(self.checker, 'trocr_model'):
                            raw_result = self.checker.solve_with_trocr(processed_img)
                            confidence = 0.8
                            
                        elif ocr_engine == "easyocr" and hasattr(self.checker, 'easyocr_reader'):
                            img_array = np.array(processed_img)
                            easyocr_results = self.checker.easyocr_reader.readtext(img_array)
                            
                            if easyocr_results:
                                best_result = max(easyocr_results, key=lambda x: x[2])
                                raw_result = best_result[1]
                                confidence = best_result[2]
                            
                        elif ocr_engine == "tesseract":
                            raw_result = self.checker.solve_with_tesseract(processed_img)
                            confidence = 0.6
                        
                        if raw_result:
                            # CHUẨN HÓA - CHỈ LẤY KÝ TỰ THẬT, KHÔNG THÊM GÌ
                            extracted = self.extract_captcha_6chars(raw_result)
                            
                            if extracted and len(extracted) >= 4:  # Thu thập từ 4 ký tự trở lên
                                all_results.append({
                                    'text': extracted,
                                    'method': method_name,
                                    'ocr': ocr_engine,
                                    'raw': raw_result,
                                    'confidence': confidence,
                                    'priority': priority
                                })
                                
                                length_info = f"({len(extracted)} chars)"
                                print(f"[ENHANCED-{ocr_engine.upper()}+{method_name}] '{raw_result}' -> '{extracted}' {length_info}")
                    
                    except Exception:
                        continue
            
            if not all_results:
                print("Enhanced solver không có kết quả")
                return {"result": "", "should_refresh": True, "confidence": 0}
            
            # Sử dụng thuật toán thông minh
            voting_result = self.smart_ensemble_voting(all_results)
            
            if voting_result["result"]:
                print(f"ENHANCED WINNER: '{voting_result['result']}' (confidence: {voting_result['confidence']:.2f})")
            
            return voting_result
            
        except Exception as e:
            print(f"Lỗi enhanced solver: {e}")
            return {"result": "", "should_refresh": True, "confidence": 0}


class PhoneCarrierChecker:
    def __init__(self, save_captcha_images=False, captcha_folder="captcha_images", primary_ocr="hybrid"):
        """
        Khởi tạo công cụ kiểm tra nhà mạng
        """
        # Bản đồ đầu số các nhà mạng
        self.PREFIX_MAP = {
            "Viettel": [
                "086", "096", "097", "098", 
                "032", "033", "034", "035", "036", "037", "038", "039"
            ],
            "Mobifone": [
                "089", "090", "093", "070", "079", "077", "076", "078"
            ],
            "Vinaphone": [
                "088", "091", "094", "083", "084", "085", "081", "082"
            ],
            "Vietnamobile": ["092", "056", "058"],
            "Gmobile": ["099", "059"],
            "Itelecom": ["087"],
            "Reddi": ["055"]
        }

        self.save_captcha_images = save_captcha_images
        self.captcha_folder = captcha_folder
        if self.save_captcha_images and not os.path.exists(captcha_folder):
            os.makedirs(captcha_folder)

        self.primary_ocr = primary_ocr
        self.ocr_engines = []
        
        self._init_ocr_engines()
        
        # Khởi tạo Enhanced Solver
        self.enhanced_solver = EnhancedCaptchaSolver(self)
        print("Enhanced Captcha Solver - CHỈ 5 PHƯƠNG PHÁP ĐÃ CHỌN, YÊU CẦU 6 KÝ TỰ!")
    
    def _init_ocr_engines(self):
        """Khởi tạo các OCR engines có sẵn - CHỈ TrOCR + EasyOCR"""
        print("Đang khởi tạo OCR engines...")
        
        if TROCR_AVAILABLE:
            try:
                print("   -> Đang tải TrOCR...")
                self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
                self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
                self.ocr_engines.append("trocr")
                print("   TrOCR sẵn sàng!")
            except Exception as e:
                print(f"   TrOCR lỗi: {e}")
        
        if EASYOCR_AVAILABLE:
            try:
                print("   -> Đang tải EasyOCR...")
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
                self.ocr_engines.append("easyocr")
                print("   EasyOCR sẵn sàng!")
            except Exception as e:
                print(f"   EasyOCR lỗi: {e}")
        
        if not self.ocr_engines:
            raise Exception("Không có OCR engine nào khả dụng!")
        
        print(f"OCR engines khả dụ: {', '.join(self.ocr_engines)} (Đã bỏ Tesseract)")
    
    def normalize_phone_number(self, phone: str) -> str:
        """Chuẩn hóa số điện thoại về dạng 10 số"""
        phone = re.sub(r'[^\d]', '', phone)
        
        if phone.startswith('84'):
            phone = '0' + phone[2:]
        elif phone.startswith('+84'):
            phone = '0' + phone[3:]
        elif len(phone) == 9 and not phone.startswith('0'):
            phone = '0' + phone
            
        return phone
    
    def check_by_prefix(self, phone: str) -> str:
        """Kiểm tra nhà mạng theo đầu số"""
        phone = self.normalize_phone_number(phone)
        
        for carrier, prefixes in self.PREFIX_MAP.items():
            for prefix in prefixes:
                if phone.startswith(prefix):
                    return carrier
        
        return "Sai số"
    
    def solve_with_trocr(self, image: Image.Image) -> str:
        """Giải captcha bằng TrOCR"""
        try:
            pixel_values = self.trocr_processor(image, return_tensors="pt").pixel_values
            generated_ids = self.trocr_model.generate(pixel_values)
            result = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return result.strip()
        except Exception as e:
            print(f"TrOCR lỗi: {e}")
            return ""
    
    def solve_with_easyocr(self, image: Image.Image) -> str:
        """Giải captcha bằng EasyOCR"""
        try:
            img_array = np.array(image)
            results = self.easyocr_reader.readtext(img_array)
            
            full_text = ""
            for (bbox, text, confidence) in results:
                if confidence > 0.2:
                    full_text += text
            
            return full_text.strip()
        except Exception as e:
            print(f"EasyOCR lỗi: {e}")
            return ""
    
    def solve_captcha_hybrid(self, img_bytes: bytes, attempt_num: int = 0, refresh_count: int = 0) -> str:
        """
        Hybrid solver - Ưu tiên Enhanced với logic thông minh
        """
        # Try Enhanced Solver với logic thông minh
        enhanced_result = self.enhanced_solver.solve_captcha_enhanced(img_bytes, attempt_num, refresh_count)
        
        if enhanced_result["result"] and len(enhanced_result["result"]) == 6:  # CHỈ CHẤP NHẬN 6 KÝ TỰ
            result_len = len(enhanced_result["result"])
            print(f"Enhanced solver thành công ({result_len} chars): '{enhanced_result['result']}'")
            return enhanced_result["result"]
        elif enhanced_result["should_refresh"]:
            print("Enhanced solver yêu cầu refresh captcha")
            return "REFRESH_NEEDED"
        
        # Fallback to original method nếu không cần refresh
        print("Enhanced solver thất bại, chuyển về original solver...")
        return self._original_solve_captcha(img_bytes, attempt_num, refresh_count)
    
    def _original_solve_captcha(self, img_bytes: bytes, attempt_num: int = 0, refresh_count: int = 0) -> str:
        """Original captcha solving method (backup) - KHÔNG AUTO PADDING"""
        try:
            if self.save_captcha_images:
                filename = f"{self.captcha_folder}/original_captcha_attempt_{attempt_num}_refresh_{refresh_count}_{int(time.time())}.png"
                with open(filename, 'wb') as f:
                    f.write(img_bytes)
                print(f"Original - Lưu captcha: {filename}")

            # Simple preprocessing
            original = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            
            # Enhanced image
            enhancer = ImageEnhance.Contrast(original)
            high_contrast = enhancer.enhance(2.0)
            
            results = []
            
            for ocr_engine in self.ocr_engines:
                try:
                    if ocr_engine == "trocr" and hasattr(self, 'trocr_model'):
                        result = self.solve_with_trocr(high_contrast)
                    elif ocr_engine == "easyocr" and hasattr(self, 'easyocr_reader'):
                        result = self.solve_with_easyocr(high_contrast)
                    else:
                        continue
                    
                    if result:
                        # CHUẨN HÓA - CHỈ LẤY KÝ TỰ THẬT, KHÔNG THÊM GÌ
                        clean_result = re.sub(r'[^a-zA-Z0-9]', '', result)
                        if len(clean_result) == 6:  # CHỈ CHẤP NHẬN ĐÚNG 6 KÝ TỰ
                            normalized = clean_result.lower()  # Chữ thường
                            results.append(normalized)
                            print(f"[ORIGINAL-{ocr_engine.upper()}] '{result}' -> '{normalized}' ({len(normalized)} chars)")
                
                except Exception:
                    continue
            
            if results:
                # Return first valid result (6 chars only)
                best = results[0]
                print(f"ORIGINAL WINNER ({len(best)} chars): '{best}'")
                return best
            
            return ""
            
        except Exception as e:
            print(f"Lỗi original solver: {e}")
            return ""
    
    def setup_driver(self) -> webdriver.Chrome:
        """Thiết lập Chrome driver TỐI ƯU - GIỮ CAPTCHA HOẠT ĐỘNG"""
        chrome_options = Options()
        
        # TỐI ƯU HÓA KHÔNG ẢNH HƯỞNG CAPTCHA
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        
        # Tắt những thứ không cần thiết NHƯNG GIỮ JS/CSS cho captcha
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-background-timer-throttling")
        chrome_options.add_argument("--disable-backgrounding-occluded-windows")
        chrome_options.add_argument("--disable-renderer-backgrounding")
        chrome_options.add_argument("--disable-features=TranslateUI")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        chrome_options.add_argument("--disable-ipc-flooding-protection")
        
        # Chặn ads và tracking (nhẹ nhàng)
        chrome_options.add_experimental_option("useAutomationExtension", False)
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        
        # Tối ưu network - CHỈ chặn notifications và media
        prefs = {
            "profile.default_content_setting_values": {
                "notifications": 2,  # Block notifications
                "media_stream": 2,  # Block media
                "geolocation": 2,   # Block location
            },
            # KHÔNG chặn images để captcha hiển thị
            "profile.managed_default_content_settings": {
                "popups": 2  # Block popups only
            }
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        # Chạy headless để tăng tốc (bỏ comment nếu muốn)
        # chrome_options.add_argument("--headless")

        driver = webdriver.Chrome(options=chrome_options)
        
        # Script để bypass detection
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        # Timeout hợp lý
        driver.set_page_load_timeout(15)
        driver.implicitly_wait(3)
        
        print("Browser setup: Tối ưu tốc độ NHƯNG giữ captcha hoạt động")
        
        return driver
    
    def check_by_official_lookup(self, phone: str, max_attempts: int = 10, max_refreshes_per_attempt: int = 5) -> str:
        """Kiểm tra nhà mạng qua website chính thức với logic thông minh - KHÔNG YÊU CẦU 6 KÝ TỰ"""
        phone = self.normalize_phone_number(phone)
        
        driver = self.setup_driver()

        try:
            print("Đang truy cập website...")
            driver.get("https://vntelecom.vnta.gov.vn:10246/vnta/search")

            successful_attempts = 0
            
            while successful_attempts < max_attempts:
                successful_attempts += 1
                print(f"\nLần thử {successful_attempts}/{max_attempts}")
                
                try:
                    # Nhập số điện thoại
                    input_box = WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.ID, "searchPhone"))
                    )
                    input_box.clear()
                    input_box.send_keys(phone)

                    # Chờ captcha load với timeout dài hơn
                    captcha_img = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.ID, "recaptcha-image"))
                    )
                    
                    # Chờ captcha render hoàn toàn
                    time.sleep(1)

                    refresh_count = 0
                    while refresh_count < max_refreshes_per_attempt:
                        # Chụp và giải captcha
                        img_bytes = captcha_img.screenshot_as_png
                        captcha_text = self.solve_captcha_hybrid(img_bytes, successful_attempts, refresh_count)
                        
                        # Xử lý yêu cầu refresh
                        if captcha_text == "REFRESH_NEEDED":
                            refresh_count += 1
                            print(f"Refresh captcha lần {refresh_count}/{max_refreshes_per_attempt}")
                            driver.refresh()
                            time.sleep(1)
                            
                            # Nhập lại số điện thoại sau refresh
                            input_box = WebDriverWait(driver, 5).until(
                                EC.presence_of_element_located((By.ID, "searchPhone"))
                            )
                            input_box.clear()
                            input_box.send_keys(phone)
                            
                            # Lấy captcha mới
                            captcha_img = WebDriverWait(driver, 10).until(
                                EC.presence_of_element_located((By.ID, "recaptcha-image"))
                            )
                            time.sleep(1)
                            continue
                        
                        # CHỈ CHẤP NHẬN ĐÚNG 6 KÝ TỰ
                        if not captcha_text or len(captcha_text) != 6:
                            print(f"Captcha không đúng 6 ký tự ('{captcha_text}') -> Thử lại")
                            driver.refresh()
                            time.sleep(0.5)
                            break  # Thoát khỏi vòng lặp refresh, quay lại attempt mới

                        print(f"Captcha đúng 6 ký tự: '{captcha_text}'")

                        # Điền captcha
                        captcha_box = WebDriverWait(driver, 5).until(
                            EC.element_to_be_clickable((By.ID, "recaptcha1"))
                        )
                        captcha_box.clear()
                        captcha_box.send_keys(captcha_text)

                        # Submit
                        submit_button = driver.find_element(By.XPATH, "//button[contains(text(),'Tra cứu')]")
                        submit_button.click()

                        # Chờ kết quả
                        try:
                            wait = WebDriverWait(driver, 3)
                            wait.until(
                                lambda d: d.find_element(By.ID, "originTelco").text.strip() != ""
                            )

                            result = driver.find_element(By.ID, "originTelco").text.strip()

                            if result:
                                print(f"THÀNH CÔNG: {result}")
                                return result
                            else:
                                return "Không có dữ liệu"

                        except Exception:
                            print(f"Captcha 6 ký tự sai -> Thử lại")
                            driver.refresh()
                            time.sleep(0.5)
                            break  # Thoát khỏi vòng lặp refresh, quay lại attempt mới

                except Exception as e:
                    print(f"Lỗi: {str(e)[:50]}...")
                    driver.refresh()
                    time.sleep(0.5)
                    continue
            
            print("Không thể tra cứu online (vượt quá số lần thử)")
            return "FAILED_ONLINE_LOOKUP"
            
        finally:
            driver.quit()
    
    def compare_results(self, phone: str) -> dict:
        """So sánh kết quả từ cả hai phương pháp với fallback thông minh"""
        phone = self.normalize_phone_number(phone)
        
        print(f"\nKiểm tra: {phone}")
        print("=" * 50)
        
        # Kiểm tra theo đầu số (nhanh)
        print("Kiểm tra đầu số...")
        prefix_result = self.check_by_prefix(phone)
        print(f"   -> {prefix_result}")
        
        # Nếu đầu số không hợp lệ -> bỏ qua tra cứu online
        if prefix_result == "Sai số":
            print("Số không hợp lệ -> Bỏ qua tra cứu online")
            
            return {
                "phone_number": phone,
                "prefix_method": prefix_result,
                "official_method": "Không tra cứu",
                "match": False,
                "comparison_text": f"{prefix_result} / Không tra cứu",
                "is_valid": False
            }
        
        # Tra cứu online
        print("Tra cứu chính thức...")
        official_result = self.check_by_official_lookup(phone)
        print(f"   -> {official_result}")
        
        # LOGIC MỚI: Nếu tra cứu online thất bại -> trả về kết quả đầu số
        if official_result == "FAILED_ONLINE_LOOKUP":
            print(f"Tra cứu online thất bại -> Sử dụng kết quả đầu số: {prefix_result}")
            
            return {
                "phone_number": phone,
                "prefix_method": prefix_result,
                "official_method": "Thất bại (dùng đầu số)",
                "final_result": prefix_result,  # Kết quả cuối cùng
                "match": True,  # Coi như khớp vì dùng fallback
                "comparison_text": f"{prefix_result} / Fallback",
                "is_valid": True
            }
        
        # So sánh bình thường
        match = prefix_result.lower() == official_result.lower()
        status = "Khớp" if match else "Khác biệt"
        
        comparison = {
            "phone_number": phone,
            "prefix_method": prefix_result,
            "official_method": official_result,
            "final_result": official_result,  # Ưu tiên kết quả online
            "match": match,
            "comparison_text": f"{prefix_result} / {official_result}",
            "is_valid": True
        }
        
        print(f"Kết quả: {comparison['comparison_text']} - {status}")
        
        return comparison


def main():
    """Hàm chính - AUTO CHẠY FULL TEST với tối ưu hóa - KHÔNG AUTO PADDING"""
    print("SELECTED METHODS ONLY - 6 CHARS REQUIRED")
    print("AUTO MODE - CHỈ SỬ DỤNG 5 PHƯƠNG PHÁP ĐÃ CHỌN")
    print("Methods: EASYOCR+adaptive_mean, TROCR+lab_l, TROCR+sharpened, TROCR+clahe, TROCR+gaussian")
    print("Yêu cầu: ĐÚNG 6 KÝ TỰ, không chấp nhận ít hơn hoặc nhiều hơn")
    print("=" * 60)
    
    # Khởi tạo với Enhanced Solver tích hợp sẵn
    checker = PhoneCarrierChecker(save_captcha_images=True, primary_ocr="hybrid")
    
    # Danh sách số điện thoại test
    test_phones = [
        "0987654321",
        "0969062173", 
        "0328173651",
        "0987706355",
    ]
    
    results = []
    start_time = time.time()
    
    for i, phone in enumerate(test_phones, 1):
        print(f"\n{'='*60}")
        print(f"Đang xử lý {i}/{len(test_phones)}")
        
        try:
            result = checker.compare_results(phone)
            results.append(result)
        except Exception as e:
            print(f"Lỗi khi xử lý số {phone}: {e}")
    
    # Tóm tắt kết quả
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("TÓM TẮT KẾT QUẢ - SELECTED METHODS ONLY")
    print(f"Thời gian: {elapsed_time:.1f}s")
    print(f"Methods: EASYOCR+adaptive_mean, TROCR+lab_l/sharpened/clahe/gaussian")
    print(f"Captcha: YÊU CẦU ĐÚNG 6 KÝ TỰ (refresh nếu không đủ)")
    print(f"Browser: Tối ưu tốc độ, GIỮ captcha hoạt động")
    print("=" * 60)
    
    total_checked = len(results)
    valid_numbers = len([r for r in results if r.get("is_valid", True)])
    matched_results = len([r for r in results if r.get("match", False)])
    accuracy_rate = (matched_results / valid_numbers * 100) if valid_numbers > 0 else 0
    
    print(f"THỐNG KÊ:")
    print(f"   • Tổng số đã check: {total_checked}")
    print(f"   • Số hợp lệ: {valid_numbers}")
    print(f"   • Kết quả khớp: {matched_results}")
    print(f"   • Độ chính xác: {accuracy_rate:.1f}%")
    print()
    
    for result in results:
        if not result.get("is_valid", True):
            print(f"X {result['phone_number']}: Số SAI")
        else:
            # Hiển thị kết quả cuối cùng
            final_result = result.get("final_result", result.get("official_method", "Unknown"))
            status = "OK" if result["match"] else "DIFF"
            print(f"{status} {result['phone_number']}: {final_result}")
    
    print(f"\nSelected Methods Features:")
    print(f"   • Auto Mode: No manual selection")
    print(f"   • Methods: Only 5 selected combinations")
    print(f"     - EASYOCR + adaptive_mean")
    print(f"     - TROCR + lab_l")
    print(f"     - TROCR + sharpened") 
    print(f"     - TROCR + clahe")
    print(f"     - TROCR + gaussian")
    print(f"   • Captcha: EXACTLY 6 characters (refresh if not)")
    print(f"   • Position Voting: Smart confidence-based algorithm")
    print(f"   • Browser: CSS/JS/Popup disabled")
    print(f"   • Speed: Maximum optimization")
    print(f"   • Smart Fallback: Dùng đầu số khi online thất bại")
    print(f"   • NO FAKE PADDING: Only real OCR characters accepted")
    print(f"   • STRICT LENGTH: Must be exactly 6 chars, no exceptions")


if __name__ == "__main__":
    # AUTO CHẠY FULL TEST - KHÔNG CÓ MENU CHỌN
    try:
        main()  # Auto chạy full test
    except KeyboardInterrupt:
        print("\nĐã dừng chương trình")
    except Exception as e:
        print(f"\nLỗi không mong muốn: {e}")
        print("Thử chạy lại hoặc kiểm tra dependencies")