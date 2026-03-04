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

from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="m1nhb1e/captchaResolve",
    filename="trocr/model.safetensors"
)

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
    def __init__(self, checker_instance):
        self.checker = checker_instance

        self.char_corrections = {
            'l': '1', '|': 'j', '1': 'i',
            'G': '6',
            'A': 'a', 'C': 'c', 'D': 'd', 'E': 'e', 'F': 'f',
            'H': 'h', 'J': 'j', 'K': 'k', 'L': 'l', 'M': 'm',
            'N': 'n', 'P': 'p', 'R': 'r', 'T': 't', 'U': 'u',
            'V': 'v', 'W': 'w', 'X': 'x', 'Y': 'y'
        }

    def advanced_preprocessing(self, img_bytes: bytes) -> list:
        try:
            original = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_array = np.array(original)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            processed_images = []
            processed_images.append(("original", original))

            height, width = img_bgr.shape[:2]
            if width < 200:
                scale_factor = 200 / width
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                resized = cv2.resize(img_bgr, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                processed_images.append(("resized", Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))))

            gaussian = cv2.GaussianBlur(img_bgr, (3, 3), 0)
            processed_images.append(("gaussian", Image.fromarray(cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB))))

            median = cv2.medianBlur(img_bgr, 3)
            processed_images.append(("median", Image.fromarray(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))))

            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsu_rgb = cv2.cvtColor(otsu, cv2.COLOR_GRAY2RGB)
            processed_images.append(("otsu", Image.fromarray(otsu_rgb)))

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

            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            l_channel = lab[:,:,0]
            l_channel_rgb = cv2.cvtColor(l_channel, cv2.COLOR_GRAY2RGB)
            processed_images.append(("lab_l", Image.fromarray(l_channel_rgb)))

            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

            lower_dark = np.array([0, 0, 0])
            upper_dark = np.array([180, 255, 100])
            mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
            result_dark = cv2.bitwise_and(gray, gray, mask=mask_dark)
            result_dark_rgb = cv2.cvtColor(result_dark, cv2.COLOR_GRAY2RGB)
            processed_images.append(("masked_dark", Image.fromarray(result_dark_rgb)))

            lower_medium = np.array([0, 50, 50])
            upper_medium = np.array([180, 255, 200])
            mask_medium = cv2.inRange(hsv, lower_medium, upper_medium)
            result_medium = cv2.bitwise_and(gray, gray, mask=mask_medium)
            result_medium_rgb = cv2.cvtColor(result_medium, cv2.COLOR_GRAY2RGB)
            processed_images.append(("masked_medium", Image.fromarray(result_medium_rgb)))

            kernel_small = np.ones((2,2), np.uint8)

            closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel_small)
            closed_rgb = cv2.cvtColor(closed, cv2.COLOR_GRAY2RGB)
            processed_images.append(("morphed_close", Image.fromarray(closed_rgb)))

            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_small)
            opened_rgb = cv2.cvtColor(opened, cv2.COLOR_GRAY2RGB)
            processed_images.append(("morphed_open", Image.fromarray(opened_rgb)))

            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = np.uint8(np.absolute(laplacian))
            sharpened = cv2.addWeighted(gray, 1.5, laplacian, -0.5, 0)
            sharpened_rgb = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
            processed_images.append(("sharpened", Image.fromarray(sharpened_rgb)))

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            clahe_img = clahe.apply(gray)
            clahe_rgb = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
            processed_images.append(("clahe", Image.fromarray(clahe_rgb)))

            return processed_images

        except Exception as e:
            print(f"Lỗi advanced preprocessing: {e}")
            return [("original", Image.open(io.BytesIO(img_bytes)).convert("RGB"))]

    def correct_characters(self, text: str) -> str:
        corrected = ""
        for char in text:
            corrected += self.char_corrections.get(char, char)
        return corrected

    def extract_captcha_6chars(self, text: str) -> str:
        if not text:
            return ""
        chars = []
        for c in text:
            if c.isalnum():
                chars.append(c)
                if len(chars) == 6:
                    break
        return ''.join(chars).lower()

    def smart_ensemble_voting(self, results: list) -> dict:
        if not results:
            return {"result": "", "should_refresh": True, "confidence": 0}

        results_6chars = [r for r in results if len(r.get('text', '')) == 6]
        results_other = [r for r in results if len(r.get('text', '')) != 6]

        total_results = len(results)

        print(f"Tổng kết quả: {total_results}")
        print(f"  - 6 ký tự: {len(results_6chars)}")
        print(f"  - Khác (bỏ qua): {len(results_other)}")

        if not results_6chars:
            print("Không có kết quả nào đúng 6 ký tự -> Refresh captcha")
            return {"result": "", "should_refresh": True, "confidence": 0}

        print("Phân tích xác suất theo vị trí ký tự (chỉ kết quả 6 ký tự)...")

        position_counts = [{} for _ in range(6)]

        for result in results_6chars:
            text = result['text']
            weight = self._get_result_weight(result)

            for pos, char in enumerate(text):
                if pos < 6:
                    if char not in position_counts[pos]:
                        position_counts[pos][char] = 0
                    position_counts[pos][char] += weight

        final_result = ""
        total_confidence = 0
        position_confidences = []

        for pos in range(6):
            if position_counts[pos]:
                sorted_chars = sorted(position_counts[pos].items(), key=lambda x: x[1], reverse=True)
                best_char = sorted_chars[0][0]
                best_count = sorted_chars[0][1]
                total_votes = sum(position_counts[pos].values())

                pos_confidence = best_count / total_votes if total_votes > 0 else 0
                position_confidences.append(pos_confidence)

                final_result += best_char

                print(f"  Vị trí {pos+1}: '{best_char}' ({best_count}/{total_votes} = {pos_confidence:.2f})")
            else:
                print(f"  Vị trí {pos+1}: Không có dữ liệu -> Refresh")
                return {"result": "", "should_refresh": True, "confidence": 0}

        if position_confidences:
            total_confidence = sum(position_confidences) / len(position_confidences)

        print(f"Kết quả được tổng hợp: '{final_result}' (confidence: {total_confidence:.2f})")

        if len(final_result) != 6:
            print(f"Kết quả cuối cùng không đúng 6 ký tự ({len(final_result)}) -> Refresh")
            return {"result": "", "should_refresh": True, "confidence": total_confidence}

        print(f"Kết quả cuối cùng đúng 6 ký tự: '{final_result}' -> OK")
        return {"result": final_result, "should_refresh": False, "confidence": total_confidence}

    def _get_result_weight(self, result: dict) -> float:
        weight = 1.0

        if result['ocr'] == 'trocr':
            weight *= 1.5
        elif result['ocr'] == 'easyocr':
            weight *= 1.2

        if result['method'] in ['lab_l', 'gaussian', 'sharpened']:
            weight *= 1.3
        elif result['method'] in ['otsu', 'adaptive_gaussian', 'clahe']:
            weight *= 1.1

        if 'confidence' in result and result['confidence']:
            weight *= (0.5 + result['confidence'])

        return weight

    def solve_captcha_enhanced(self, img_bytes: bytes, attempt_num: int = 0, refresh_count: int = 0) -> dict:
        try:
            if self.checker.save_captcha_images:
                filename = f"{self.checker.captcha_folder}/enhanced_captcha_attempt_{attempt_num}_refresh_{refresh_count}_{int(time.time())}.png"
                with open(filename, 'wb') as f:
                    f.write(img_bytes)
                print(f"Lưu captcha: {filename}")

            processed_images = self.advanced_preprocessing(img_bytes)
            all_results = []

            selected_combinations = []

            for method_name, processed_img in processed_images:
                for ocr_engine in self.checker.ocr_engines:
                    valid_method = False

                    if ocr_engine == "easyocr" and method_name == "adaptive_mean":
                        valid_method = True
                    elif ocr_engine == "trocr" and method_name in ["lab_l", "sharpened", "clahe", "gaussian"]:
                        valid_method = True

                    if valid_method:
                        selected_combinations.append((method_name, processed_img, ocr_engine, "SELECTED"))

            all_combinations = selected_combinations

            if not all_combinations:
                print("Không có combination nào được chọn -> Fallback to original")
                for method_name, processed_img in processed_images[:3]:
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
                            extracted = self.extract_captcha_6chars(raw_result)

                            if extracted and len(extracted) >= 4:
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

            voting_result = self.smart_ensemble_voting(all_results)

            if voting_result["result"]:
                print(f"ENHANCED WINNER: '{voting_result['result']}' (confidence: {voting_result['confidence']:.2f})")

            return voting_result

        except Exception as e:
            print(f"Lỗi enhanced solver: {e}")
            return {"result": "", "should_refresh": True, "confidence": 0}


class PhoneCarrierChecker:
    def __init__(self, save_captcha_images=False, captcha_folder="captcha_images", primary_ocr="hybrid"):
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

        self.enhanced_solver = EnhancedCaptchaSolver(self)
        print("Enhanced Captcha Solver - CHỈ 5 PHƯƠNG PHÁP ĐÃ CHỌN, YÊU CẦU 6 KÝ TỰ!")

    def _init_ocr_engines(self):
        print("Đang khởi tạo OCR engines...")

        if TROCR_AVAILABLE:
            try:
                print("   -> Đang tải Custom TrOCR...")
                self.trocr_processor = TrOCRProcessor.from_pretrained("m1nhb1e/captchaResolve")
                self.trocr_model = VisionEncoderDecoderModel.from_pretrained("m1nhb1e/captchaResolve")
                self.ocr_engines.append("trocr")
                print("   Custom TrOCR sẵn sàng!")
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

        print(f"OCR engines khả dụ: {', '.join(self.ocr_engines)}")

    def normalize_phone_number(self, phone: str) -> str:
        phone = re.sub(r'[^\d]', '', phone)

        if phone.startswith('84'):
            phone = '0' + phone[2:]
        elif phone.startswith('+84'):
            phone = '0' + phone[3:]
        elif len(phone) == 9 and not phone.startswith('0'):
            phone = '0' + phone

        return phone

    def check_by_prefix(self, phone: str) -> str:
        phone = self.normalize_phone_number(phone)

        for carrier, prefixes in self.PREFIX_MAP.items():
            for prefix in prefixes:
                if phone.startswith(prefix):
                    return carrier

        return "Sai số"

    def solve_with_trocr(self, image: Image.Image) -> str:
        try:
            pixel_values = self.trocr_processor(image, return_tensors="pt").pixel_values
            generated_ids = self.trocr_model.generate(pixel_values)
            result = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return result.strip()
        except Exception as e:
            print(f"TrOCR lỗi: {e}")
            return ""

    def solve_with_easyocr(self, image: Image.Image) -> str:
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
        enhanced_result = self.enhanced_solver.solve_captcha_enhanced(img_bytes, attempt_num, refresh_count)

        if enhanced_result["result"] and len(enhanced_result["result"]) == 6:
            result_len = len(enhanced_result["result"])
            print(f"Enhanced solver thành công ({result_len} chars): '{enhanced_result['result']}'")
            return enhanced_result["result"]
        elif enhanced_result["should_refresh"]:
            print("Enhanced solver yêu cầu refresh captcha")
            return "REFRESH_NEEDED"

        print("Enhanced solver thất bại, chuyển về original solver...")
        return self._original_solve_captcha(img_bytes, attempt_num, refresh_count)

    def _original_solve_captcha(self, img_bytes: bytes, attempt_num: int = 0, refresh_count: int = 0) -> str:
        try:
            if self.save_captcha_images:
                filename = f"{self.captcha_folder}/original_captcha_attempt_{attempt_num}_refresh_{refresh_count}_{int(time.time())}.png"
                with open(filename, 'wb') as f:
                    f.write(img_bytes)
                print(f"Original - Lưu captcha: {filename}")

            original = Image.open(io.BytesIO(img_bytes)).convert("RGB")

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
                        clean_result = re.sub(r'[^a-zA-Z0-9]', '', result)
                        if len(clean_result) == 6:
                            normalized = clean_result.lower()
                            results.append(normalized)
                            print(f"[ORIGINAL-{ocr_engine.upper()}] '{result}' -> '{normalized}' ({len(normalized)} chars)")

                except Exception:
                    continue

            if results:
                best = results[0]
                print(f"ORIGINAL WINNER ({len(best)} chars): '{best}'")
                return best

            return ""

        except Exception as e:
            print(f"Lỗi original solver: {e}")
            return ""

    def setup_driver(self) -> webdriver.Chrome:
        chrome_options = Options()

        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-background-timer-throttling")
        chrome_options.add_argument("--disable-backgrounding-occluded-windows")
        chrome_options.add_argument("--disable-renderer-backgrounding")
        chrome_options.add_argument("--disable-features=TranslateUI")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        chrome_options.add_argument("--disable-ipc-flooding-protection")

        chrome_options.add_experimental_option("useAutomationExtension", False)
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])

        prefs = {
            "profile.default_content_setting_values": {
                "notifications": 2,
                "media_stream": 2,
                "geolocation": 2,
            },
            "profile.managed_default_content_settings": {
                "popups": 2
            }
        }
        chrome_options.add_experimental_option("prefs", prefs)

        driver = webdriver.Chrome(options=chrome_options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        driver.set_page_load_timeout(15)
        driver.implicitly_wait(3)

        print("Browser setup hoàn tất")

        return driver

    def check_by_official_lookup(self, phone: str, max_attempts: int = 10, max_refreshes_per_attempt: int = 5) -> str:
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
                    input_box = WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.ID, "searchPhone"))
                    )
                    input_box.clear()
                    input_box.send_keys(phone)

                    captcha_img = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.ID, "recaptcha-image"))
                    )

                    time.sleep(1)

                    refresh_count = 0
                    while refresh_count < max_refreshes_per_attempt:
                        img_bytes = captcha_img.screenshot_as_png
                        captcha_text = self.solve_captcha_hybrid(img_bytes, successful_attempts, refresh_count)

                        if captcha_text == "REFRESH_NEEDED":
                            refresh_count += 1
                            print(f"Refresh captcha lần {refresh_count}/{max_refreshes_per_attempt}")
                            driver.refresh()
                            time.sleep(1)

                            input_box = WebDriverWait(driver, 5).until(
                                EC.presence_of_element_located((By.ID, "searchPhone"))
                            )
                            input_box.clear()
                            input_box.send_keys(phone)

                            captcha_img = WebDriverWait(driver, 10).until(
                                EC.presence_of_element_located((By.ID, "recaptcha-image"))
                            )
                            time.sleep(1)
                            continue

                        if not captcha_text or len(captcha_text) != 6:
                            print(f"Captcha không đúng 6 ký tự ('{captcha_text}') -> Thử lại")
                            driver.refresh()
                            time.sleep(0.5)
                            break

                        print(f"Captcha đúng 6 ký tự: '{captcha_text}'")

                        captcha_box = WebDriverWait(driver, 5).until(
                            EC.element_to_be_clickable((By.ID, "recaptcha1"))
                        )
                        captcha_box.clear()
                        captcha_box.send_keys(captcha_text)

                        submit_button = driver.find_element(By.XPATH, "//button[contains(text(),'Tra cứu')]")
                        submit_button.click()

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
                            break

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
        phone = self.normalize_phone_number(phone)

        print(f"\nKiểm tra: {phone}")
        print("=" * 50)

        print("Kiểm tra đầu số...")
        prefix_result = self.check_by_prefix(phone)
        print(f"   -> {prefix_result}")

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

        print("Tra cứu chính thức...")
        official_result = self.check_by_official_lookup(phone)
        print(f"   -> {official_result}")

        if official_result == "FAILED_ONLINE_LOOKUP":
            print(f"Tra cứu online thất bại -> Sử dụng kết quả đầu số: {prefix_result}")
            return {
                "phone_number": phone,
                "prefix_method": prefix_result,
                "official_method": "Thất bại (dùng đầu số)",
                "final_result": prefix_result,
                "match": True,
                "comparison_text": f"{prefix_result} / Fallback",
                "is_valid": True
            }

        match = prefix_result.lower() == official_result.lower()
        status = "Khớp" if match else "Khác biệt"

        comparison = {
            "phone_number": phone,
            "prefix_method": prefix_result,
            "official_method": official_result,
            "final_result": official_result,
            "match": match,
            "comparison_text": f"{prefix_result} / {official_result}",
            "is_valid": True
        }

        print(f"Kết quả: {comparison['comparison_text']} - {status}")

        return comparison


def main():
    print("CUSTOM TROCR + EASYOCR - 6 CHARS REQUIRED")
    print("=" * 60)

    checker = PhoneCarrierChecker(save_captcha_images=True, primary_ocr="hybrid")

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

    elapsed_time = time.time() - start_time
    total_checked = len(results)
    valid_numbers = len([r for r in results if r.get("is_valid", True)])
    matched_results = len([r for r in results if r.get("match", False)])
    accuracy_rate = (matched_results / valid_numbers * 100) if valid_numbers > 0 else 0

    print(f"\n{'='*60}")
    print("TÓM TẮT KẾT QUẢ")
    print(f"Thời gian: {elapsed_time:.1f}s")
    print(f"Tổng: {total_checked} | Hợp lệ: {valid_numbers} | Khớp: {matched_results} | Độ chính xác: {accuracy_rate:.1f}%")
    print("-" * 60)

    for result in results:
        if not result.get("is_valid", True):
            print(f"X {result['phone_number']}: Số SAI")
        else:
            final_result = result.get("final_result", result.get("official_method", "Unknown"))
            status = "OK" if result["match"] else "DIFF"
            print(f"{status} {result['phone_number']}: {final_result}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nĐã dừng chương trình")
    except Exception as e:
        print(f"\nLỗi không mong muốn: {e}")
        print("Thử chạy lại hoặc kiểm tra dependencies")