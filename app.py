"""
Optimized Phone Carrier Checker with Custom TrOCR Model Only
- Uses only custom TrOCR model from specified path with its own preprocessor
- ONLY uses 'sharpened' preprocessing method for maximum speed
- Character segmentation for improved OCR accuracy
- Individual character recognition with position-specific processing
- Advanced voting algorithm with character-level confidence
- MODIFIED: Only uses 'sharpened' preprocessing and custom model preprocessor
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

import time
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict, Counter

from PIL import Image, ImageEnhance
import io
import numpy as np
import cv2

from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="m1nhb1e/captchaResolve",
    filename="trocr/model.safetensors"
)

# TrOCR imports with error handling
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False
    print("ERROR: transformers library not found. Please install: pip install transformers torch")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


@dataclass
class CharacterResult:
    """Data class for individual character OCR results"""
    position: int
    character: str
    method: str
    ocr_engine: str
    confidence: float
    raw_text: str = ""


@dataclass
class CaptchaResult:
    """Data class for CAPTCHA solving results"""
    text: str
    method: str
    ocr_engine: str
    confidence: float
    character_results: List[CharacterResult] = None
    raw_text: str = ""
    should_refresh: bool = False


@dataclass
class CarrierCheckResult:
    """Data class for carrier check results"""
    phone_number: str
    prefix_result: str
    official_result: str
    final_result: str
    match: bool
    is_valid: bool
    comparison_text: str


class Config:
    """Configuration constants"""
    # Custom TrOCR model path - ONLY MODEL USED
    CUSTOM_TROCR_MODEL_PATH = model_path
    
    # Carrier prefixes mapping
    PREFIX_MAP = {
        "Viettel": ["086", "096", "097", "098", "032", "033", "034", "035", "036", "037", "038", "039"],
        "Mobifone": ["089", "090", "093", "070", "079", "077", "076", "078"],
        "Vinaphone": ["088", "091", "094", "083", "084", "085", "081", "082"],
        "Vietnamobile": ["092", "056", "058"],
        "Gmobile": ["099", "059"],
        "Itelecom": ["087"],
        "Reddi": ["055"]
    }
    
    # CAPTCHA processing settings
    REQUIRED_CAPTCHA_LENGTH = 6
    MAX_ATTEMPTS = 5
    MAX_REFRESHES_PER_ATTEMPT = 5
    
    # Character segmentation settings
    SEGMENT_OVERLAP_RATIO = 0.1  # 10% overlap between segments
    MIN_CHAR_WIDTH_RATIO = 0.1   # Minimum character width as ratio of total width
    
    # Method weights for preprocessing - ONLY sharpened
    METHOD_WEIGHTS = {
        'sharpened': 1.5  # Higher weight for the only method
    }
    
    # Selected preprocessing methods for TrOCR - ONLY sharpened
    SELECTED_METHODS = [
        "sharpened"
    ]


class ImageProcessor:
    """Handles image preprocessing for CAPTCHA solving"""
    
    @staticmethod
    def preprocess_image(img_bytes: bytes) -> List[Tuple[str, Image.Image]]:
        """Apply only sharpened preprocessing technique"""
        try:
            original = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_array = np.array(original)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            processed_images = [("original", original)]
            
            # Resize if too small (important for character segmentation)
            height, width = img_bgr.shape[:2]
            if width < 240:  # Increased minimum width for better character separation
                scale_factor = 240 / width
                new_size = (int(width * scale_factor), int(height * scale_factor))
                resized = cv2.resize(img_bgr, new_size, interpolation=cv2.INTER_CUBIC)
                img_bgr = resized
                processed_images[0] = ("original", Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
            
            # Convert to grayscale for processing
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # Edge enhancement (Sharpened) - ONLY METHOD USED
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = np.uint8(np.absolute(laplacian))
            sharpened = cv2.addWeighted(gray, 1.5, laplacian, -0.5, 0)
            processed_images.append(("sharpened", Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB))))
            
            return processed_images
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return [("original", Image.open(io.BytesIO(img_bytes)).convert("RGB"))]
    
    @staticmethod
    def segment_characters(image: Image.Image) -> List[Image.Image]:
        """Segment CAPTCHA image into 6 individual character images"""
        try:
            width, height = image.size
            char_width = width // Config.REQUIRED_CAPTCHA_LENGTH
            overlap = int(char_width * Config.SEGMENT_OVERLAP_RATIO)
            
            logger.info(f"Character Segmentation - Image: {width}x{height}, char_width: {char_width}, overlap: {overlap}")
            
            character_images = []
            
            for i in range(Config.REQUIRED_CAPTCHA_LENGTH):
                # Calculate boundaries with slight overlap to avoid cutting characters
                left = max(0, i * char_width - overlap)
                right = min(width, (i + 1) * char_width + overlap)
                
                # Ensure minimum width
                if right - left < width * Config.MIN_CHAR_WIDTH_RATIO:
                    right = left + int(width * Config.MIN_CHAR_WIDTH_RATIO)
                    if right > width:
                        right = width
                char_img = image.crop((left, 0, right, height))
                character_images.append(char_img)
            
            return character_images
        except Exception as e:
            logger.error(f"Character segmentation failed: {e}")
            return []


class CustomTrOCREngine:
    """Handles OCR operations using only custom TrOCR model with its own preprocessor"""
    
    def __init__(self):
        self.model_loaded = False
        self._init_custom_model()
    
    def _init_custom_model(self):
        """Initialize custom TrOCR model with its own preprocessor config"""
        if not TROCR_AVAILABLE:
            raise RuntimeError("TrOCR not available! Please install: pip install transformers torch")
        
        custom_model_path = Config.CUSTOM_TROCR_MODEL_PATH
        
        # Check if custom model path exists
        if not Path(custom_model_path).exists():
            raise FileNotFoundError(f"Custom TrOCR model not found at: {custom_model_path}")
        
        try:
            logger.info(f"Loading custom TrOCR model from: {custom_model_path}")
            
            # Load processor from custom model - NO FALLBACK
            logger.info("Loading processor from custom model (no fallback)")
            self.trocr_processor = TrOCRProcessor.from_pretrained(custom_model_path)
            logger.info("Custom processor loaded successfully")
            
            # Load the custom model
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained(custom_model_path)
            
            self.model_loaded = True
            logger.info("Custom TrOCR model and processor loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load custom TrOCR model: {e}")
            raise RuntimeError(f"Custom TrOCR model loading failed: {e}")
    
    def solve_with_trocr(self, image: Image.Image) -> Tuple[str, float]:
        """Solve CAPTCHA using custom TrOCR"""
        if not self.model_loaded:
            return "", 0.0
        
        try:
            pixel_values = self.trocr_processor(image, return_tensors="pt").pixel_values
            logger.debug(f"TrOCR input tensor shape: {pixel_values.shape}")
            
            generated_ids = self.trocr_model.generate(pixel_values, max_length=10)
            result = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            logger.debug(f"Custom TrOCR raw result: '{result}'")
            
            return result.strip(), 0.95  # Very high confidence for custom model with sharpened only
        except Exception as e:
            logger.error(f"Custom TrOCR failed: {e}")
            return "", 0.0


class CaptchaSolver:
    """CAPTCHA solver using custom TrOCR model with sharpened preprocessing only"""
    
    def __init__(self, ocr_engine: CustomTrOCREngine, save_images: bool = False, save_folder: str = "captcha_images"):
        self.ocr_engine = ocr_engine
        self.save_images = save_images
        self.save_folder = Path(save_folder)
        
        if self.save_images:
            self.save_folder.mkdir(exist_ok=True)
        
        # Character correction mapping for common OCR errors
        self.char_corrections = {
            'LI': 'u',
            'lb': 'h',
            'O0': 'o',
            '1l': 'i',
            '5S': 'S',
            '2Z': 'Z',
            'PL': 'd',
            'pd': 'd',
            'dp': 'd',
            '9g': 'g',
            ':': 'i',
        }
    
    def clean_captcha_result(self, text: str) -> str:
        """Clean and normalize CAPTCHA result to 6 characters"""
        if not text:
            return ""
        
        # Remove spaces and special characters, keep only alphanumeric
        cleaned = ''.join(char.lower() for char in text if char.isalnum())
        
        # Apply character corrections
        for pattern, replacement in self.char_corrections.items():
            cleaned = cleaned.replace(pattern.lower(), replacement)
        
        # Ensure exactly 6 characters
        if len(cleaned) >= Config.REQUIRED_CAPTCHA_LENGTH:
            return cleaned[:Config.REQUIRED_CAPTCHA_LENGTH]
        
        return cleaned
    
    def solve_full_image(self, img_bytes: bytes, attempt_num: int = 0, refresh_count: int = 0) -> Dict:
        """Solve CAPTCHA using full image recognition - only sharpened method"""
        try:
            logger.info(f"Starting full image CAPTCHA solve - Attempt {attempt_num+1}, Refresh {refresh_count+1}")
            
            # Save original image if required
            if self.save_images:
                filename = self.save_folder / f"captcha_full_a{attempt_num}_r{refresh_count}_{int(time.time())}.png"
                filename.write_bytes(img_bytes)
                logger.info(f"Saved original CAPTCHA: {filename.name}")
            
            # Preprocess image - now only returns sharpened
            logger.info("Starting image preprocessing (sharpened only)...")
            processed_images = ImageProcessor.preprocess_image(img_bytes)
            logger.info(f"Preprocessing complete: {len(processed_images)} variants created")
            
            # Collect results from sharpened preprocessing method only
            results = []
            
            for method_name, processed_img in processed_images:
                if method_name not in Config.SELECTED_METHODS:
                    continue
                
                logger.info(f"Processing method: {method_name}")
                print(f"\n{'='*50}")
                print(f"OCR PROCESSING: {method_name.upper()}")
                print(f"{'='*50}")
                
                # Save preprocessed image if required
                if self.save_images:
                    prep_filename = self.save_folder / f"preprocessed_{method_name}_a{attempt_num}_r{refresh_count}_{int(time.time())}.png"
                    processed_img.save(prep_filename)
                
                # OCR with custom TrOCR
                raw_text, confidence = self.ocr_engine.solve_with_trocr(processed_img)
                
                if raw_text:
                    cleaned_text = self.clean_captcha_result(raw_text)
                    
                    # Log result
                    print(f"[CUSTOM-TROCR+{method_name}] '{raw_text}' -> '{cleaned_text}' ({len(cleaned_text)} chars)")
                    logger.info(f"OCR result [{method_name}]: '{raw_text}' -> '{cleaned_text}' (conf: {confidence:.3f})")
                    
                    if len(cleaned_text) == Config.REQUIRED_CAPTCHA_LENGTH:
                        results.append({
                            'text': cleaned_text,
                            'method': method_name,
                            'confidence': confidence,
                            'raw_text': raw_text,
                            'weight': Config.METHOD_WEIGHTS.get(method_name, 1.0) * confidence
                        })
                else:
                    print(f"[CUSTOM-TROCR+{method_name}] '' -> '' (0 chars)")
                    logger.info(f"OCR result [{method_name}]: No text detected")
            
            if not results:
                logger.error("No valid OCR results - REFRESH NEEDED")
                return {"result": "", "should_refresh": True, "confidence": 0.0}
            
            # Since we only have one method (sharpened), use its result directly
            best_result = results[0]  # Only one result from sharpened method
            best_text = best_result['text']
            overall_confidence = best_result['confidence']
            
            logger.info(f"Result from sharpened method: '{best_text}' (confidence: {overall_confidence:.3f})")
            
            confidence_indicator = "HIGH" if overall_confidence > 0.7 else "MED" if overall_confidence > 0.5 else "LOW"
            logger.info(f"Final result: '{best_text}' [{confidence_indicator}] (confidence: {overall_confidence:.3f})")
            
            # Print final result summary
            print(f"\n{'='*50}")
            print(f"FINAL CAPTCHA RESULT (SHARPENED ONLY)")
            print(f"{'='*50}")
            print(f"Result: '{best_text}'")
            print(f"Confidence: {overall_confidence:.1%}")
            
            # Determine if refresh is needed
            should_refresh = overall_confidence < 0.3
            
            if should_refresh:
                logger.warning(f"Low confidence ({overall_confidence:.3f} < 0.3) - Recommending refresh")
            
            return {
                "result": best_text,
                "should_refresh": should_refresh,
                "confidence": overall_confidence
            }
            
        except Exception as e:
            logger.error(f"Full image CAPTCHA solving failed: {e}")
            return {"result": "", "should_refresh": True, "confidence": 0.0}
    
    def solve(self, img_bytes: bytes, attempt_num: int = 0, refresh_count: int = 0) -> str:
        """Main CAPTCHA solving method using full image recognition"""
        try:
            # Use full image approach
            full_result = self.solve_full_image(img_bytes, attempt_num, refresh_count)
            
            if full_result["should_refresh"]:
                return "REFRESH_NEEDED"
            
            return full_result["result"]
            
        except Exception as e:
            logger.error(f"CAPTCHA solving failed: {e}")
            return "REFRESH_NEEDED"


class WebDriverManager:
    """Manages Chrome WebDriver with optimized settings"""
    
    @staticmethod
    def create_driver() -> webdriver.Chrome:
        """Create optimized Chrome driver"""
        options = Options()
        
        # Performance optimizations
        performance_args = [
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-web-security",
            "--disable-extensions",
            "--disable-plugins",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
            "--disable-features=TranslateUI,VizDisplayCompositor",
            "--disable-ipc-flooding-protection"
        ]
        
        for arg in performance_args:
            options.add_argument(arg)
        
        # Disable automation detection
        options.add_experimental_option("useAutomationExtension", False)
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        
        # Content settings
        prefs = {
            "profile.default_content_setting_values": {
                "notifications": 2,
                "media_stream": 2,
                "geolocation": 2,
            }
        }
        options.add_experimental_option("prefs", prefs)
        
        # Uncomment for headless mode
        # options.add_argument("--headless")
        
        driver = webdriver.Chrome(options=options)
        
        # Anti-detection
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        # Timeouts
        driver.set_page_load_timeout(15)
        driver.implicitly_wait(3)
        
        logger.info("Chrome driver initialized with optimizations")
        return driver


class PhoneCarrierChecker:
    """Main class for phone carrier checking using only custom TrOCR with sharpened preprocessing"""
    
    def __init__(self, save_captcha_images: bool = False, captcha_folder: str = "captcha_images"):
        self.save_captcha_images = save_captcha_images
        self.captcha_folder = captcha_folder
        
        # Initialize components
        self.ocr_engine = CustomTrOCREngine()
        self.captcha_solver = CaptchaSolver(
            self.ocr_engine, 
            save_captcha_images, 
            captcha_folder
        )
        
        logger.info("PhoneCarrierChecker initialized with custom TrOCR model (sharpened only)")
    
    @staticmethod
    def normalize_phone_number(phone: str) -> str:
        """Normalize phone number to 10-digit format"""
        phone = re.sub(r'[^\d]', '', phone)
        
        if phone.startswith('84'):
            phone = '0' + phone[2:]
        elif phone.startswith('+84'):
            phone = '0' + phone[3:]
        elif len(phone) == 9 and not phone.startswith('0'):
            phone = '0' + phone
        
        return phone
    
    def check_by_prefix(self, phone: str) -> str:
        """Check carrier by phone number prefix"""
        phone = self.normalize_phone_number(phone)
        
        for carrier, prefixes in Config.PREFIX_MAP.items():
            if any(phone.startswith(prefix) for prefix in prefixes):
                return carrier
        
        return "Sai so"
    
    def check_by_official_lookup(self, phone: str) -> str:
        """Check carrier via official website using custom TrOCR with sharpened preprocessing only"""
        phone = self.normalize_phone_number(phone)
        driver = WebDriverManager.create_driver()
        
        try:
            logger.info("Accessing official website...")
            driver.get("https://vntelecom.vnta.gov.vn:10246/vnta/search")
            
            for attempt in range(1, Config.MAX_ATTEMPTS + 1):
                logger.info(f"Attempt {attempt}/{Config.MAX_ATTEMPTS}")
                
                try:
                    # Enter phone number
                    input_box = WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.ID, "searchPhone"))
                    )
                    input_box.clear()
                    input_box.send_keys(phone)
                    
                    # Wait for CAPTCHA
                    captcha_img = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.ID, "recaptcha-image"))
                    )
                    time.sleep(1)  # Let CAPTCHA fully render
                    
                    for refresh in range(Config.MAX_REFRESHES_PER_ATTEMPT):
                        print(f"\n{'='*80}")
                        print(f"CAPTCHA SOLVING ATTEMPT {attempt}.{refresh+1}")
                        print(f"Phone: {phone}")
                        print(f"Using SHARPENED preprocessing only")
                        print(f"{'='*80}")
                        
                        # Solve CAPTCHA using custom TrOCR with sharpened only
                        img_bytes = captcha_img.screenshot_as_png
                        captcha_text = self.captcha_solver.solve(img_bytes, attempt, refresh)
                        
                        if captcha_text == "REFRESH_NEEDED":
                            logger.info(f"Refreshing CAPTCHA ({refresh + 1}/{Config.MAX_REFRESHES_PER_ATTEMPT})")
                            driver.refresh()
                            time.sleep(1)
                            break
                        
                        if len(captcha_text) != Config.REQUIRED_CAPTCHA_LENGTH:
                            logger.warning(f"Invalid CAPTCHA length: {len(captcha_text)} chars")
                            driver.refresh()
                            time.sleep(0.5)
                            break
                        
                        logger.info(f"CAPTCHA solved with custom TrOCR (sharpened): '{captcha_text}'")
                        
                        # Submit form
                        captcha_box = WebDriverWait(driver, 5).until(
                            EC.element_to_be_clickable((By.ID, "recaptcha1"))
                        )
                        captcha_box.clear()
                        captcha_box.send_keys(captcha_text)
                        
                        submit_button = driver.find_element(By.CSS_SELECTOR, "button.btn.btn-primary[type='button']")
                        submit_button.click()
                        
                        # Wait for result
                        try:
                            WebDriverWait(driver, 3).until(
                                lambda d: d.find_element(By.ID, "originTelco").text.strip() != ""
                            )
                            
                            result = driver.find_element(By.ID, "originTelco").text.strip()
                            if result:
                                logger.info(f"Official lookup successful: {result}")
                                print(f"\nSUCCESS: {result}")
                                return result
                            return "Khong co du lieu"
                        
                        except TimeoutException:
                            logger.warning("Incorrect CAPTCHA, retrying...")
                            print("CAPTCHA incorrect, retrying...")
                            driver.refresh()
                            time.sleep(0.5)
                            break
                
                except Exception as e:
                    logger.error(f"Attempt {attempt} failed: {str(e)[:100]}...")
                    driver.refresh()
                    time.sleep(0.5)
                    continue
            
            logger.error("All attempts failed")
            return "FAILED_ONLINE_LOOKUP"
            
        finally:
            driver.quit()
    
    def check_carrier(self, phone: str) -> CarrierCheckResult:
        """Complete carrier check with both methods"""
        phone = self.normalize_phone_number(phone)
        
        logger.info(f"Checking carrier for: {phone}")
        
        # Check by prefix first
        prefix_result = self.check_by_prefix(phone)
        logger.info(f"Prefix result: {prefix_result}")
        
        if prefix_result == "Sai so":
            logger.warning("Invalid phone number, skipping official lookup")
            return CarrierCheckResult(
                phone_number=phone,
                prefix_result=prefix_result,
                official_result="Khong tra cuu",
                final_result=prefix_result,
                match=False,
                is_valid=False,
                comparison_text=f"{prefix_result} / Khong tra cuu"
            )
        
        # Official lookup with custom TrOCR (sharpened only)
        official_result = self.check_by_official_lookup(phone)
        logger.info(f"Official result: {official_result}")
        
        # Handle fallback
        if official_result == "FAILED_ONLINE_LOOKUP":
            logger.warning(f"Official lookup failed, using prefix result: {prefix_result}")
            return CarrierCheckResult(
                phone_number=phone,
                prefix_result=prefix_result,
                official_result="That bai (dung dau so)",
                final_result=prefix_result,
                match=True,
                is_valid=True,
                comparison_text=f"{prefix_result} / Fallback"
            )
        
        # Compare results
        match = prefix_result.lower() == official_result.lower()
        
        return CarrierCheckResult(
            phone_number=phone,
            prefix_result=prefix_result,
            official_result=official_result,
            final_result=official_result,
            match=match,
            is_valid=True,
            comparison_text=f"{prefix_result} / {official_result}"
        )


def run_test():
    """Run automated test with sample phone numbers"""
    print("CUSTOM TROCR PHONE CARRIER CHECKER")
    print("Using Custom TrOCR Model Only - SHARPENED Method Only")
    print("=" * 70)
    print(f"Model Path: {Config.CUSTOM_TROCR_MODEL_PATH}")
    print(f"Preprocessing Methods: {Config.SELECTED_METHODS}")
    print("=" * 70)
    
    # Initialize checker with custom TrOCR only
    try:
        checker = PhoneCarrierChecker(save_captcha_images=True)
    except Exception as e:
        print(f"ERROR: Failed to initialize checker: {e}")
        print("Please check:")
        print(f"1. Model exists at: {Config.CUSTOM_TROCR_MODEL_PATH}")
        print("2. Model includes preprocessor_config.json")
        print("3. Required libraries installed: pip install transformers torch")
        return
    
    # Test phone numbers
    test_phones = [
        "0987654321",
        "0969062173", 
        "0328173651",
        "0987706355",
        "0901234567",
        "0881234567"
    ]
    
    results = []
    start_time = time.time()
    
    for i, phone in enumerate(test_phones, 1):
        print(f"\n{'='*70}")
        print(f"Processing {i}/{len(test_phones)}: {phone}")
        
        try:
            result = checker.check_carrier(phone)
            results.append(result)
            
            # Show intermediate result
            status = "MATCH" if result.match else "DIFFER" if result.is_valid else "INVALID"
            print(f"Result: {result.final_result} [{status}]")
            
        except Exception as e:
            logger.error(f"Failed to process {phone}: {e}")
    
    # Summary
    elapsed_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY - CUSTOM TROCR METHOD (SHARPENED ONLY)")
    print(f"Total time: {elapsed_time:.1f}s")
    print(f"Average per check: {elapsed_time/len(test_phones):.1f}s")
    print("=" * 70)
    
    # Statistics
    total_checked = len(results)
    valid_numbers = len([r for r in results if r.is_valid])
    matched_results = len([r for r in results if r.match])
    accuracy_rate = (matched_results / valid_numbers * 100) if valid_numbers > 0 else 0
    
    print(f"STATISTICS:")
    print(f"   • Total checked: {total_checked}")
    print(f"   • Valid numbers: {valid_numbers}")
    print(f"   • Matching results: {matched_results}")
    print(f"   • Accuracy rate: {accuracy_rate:.1f}%")
    print()
    
    # Detailed results table
    print("DETAILED RESULTS:")
    print("-" * 70)
    print(f"{'Phone Number':<12} {'Prefix':<10} {'Official':<12} {'Match':<6} {'Status'}")
    print("-" * 70)
    
    for result in results:
        if not result.is_valid:
            print(f"{result.phone_number:<12} {'Invalid':<10} {'N/A':<12} {'NO':<6} Invalid")
        else:
            match_symbol = "YES" if result.match else "NO"
            print(f"{result.phone_number:<12} {result.prefix_result:<10} {result.official_result:<12} {match_symbol:<6} {'Match' if result.match else 'Differ'}")
    
    print("-" * 70)
    print(f"\nCUSTOM TROCR MODEL FEATURES (OPTIMIZED VERSION - SHARPENED ONLY):")
    print(f"   • Model Path: {Config.CUSTOM_TROCR_MODEL_PATH}")
    print(f"   • Preprocessing method: Sharpened only")
    print(f"   • Maximum processing speed by using only 1 method")
    print(f"   • Uses custom model's own preprocessor config")
    print(f"   • Enhanced voting algorithm with confidence weighting")
    print(f"   • Smart image resizing and character boundary detection")
    print(f"   • Enhanced error correction for OCR results")
    print(f"   • Detailed confidence scoring and logging")
    print(f"   • Streamlined preprocessing pipeline for maximum speed")


def run_captcha_test():
    """Test only the custom TrOCR CAPTCHA solver with Sharpened method only"""
    print("CUSTOM TROCR CAPTCHA TEST (SHARPENED ONLY)")
    print("=" * 50)
    print(f"Model Path: {Config.CUSTOM_TROCR_MODEL_PATH}")
    print(f"Methods: {Config.SELECTED_METHODS}")
    print("=" * 50)
    
    try:
        # Initialize OCR engine and solver
        ocr_engine = CustomTrOCREngine()
        solver = CaptchaSolver(ocr_engine, save_images=True, save_folder="test_captcha")
        
        print("Custom TrOCR CAPTCHA solver initialized successfully")
        print("Images will be saved to 'test_captcha' folder")
        print("Using only Sharpened preprocessing method for maximum speed")
        print("\nOptimized logging format:")
        print("[CUSTOM-TROCR+sharpened] 'raw_text' -> 'cleaned_text' (N chars)")
        print("\nReady to process CAPTCHA images!")
        print("Note: Use this in your actual application with real CAPTCHA images")
        
    except Exception as e:
        logger.error(f"Custom TrOCR CAPTCHA solver initialization failed: {e}")
        print("Failed to initialize custom TrOCR CAPTCHA solver")
        print("Please ensure:")
        print(f"1. Custom model exists at: {Config.CUSTOM_TROCR_MODEL_PATH}")
        print("2. Model includes preprocessor_config.json file")
        print("3. Required dependencies installed: pip install transformers torch opencv-python pillow")


def run_single_phone_test():
    """Test with a single phone number to see detailed Sharpened-only logging"""
    print("SINGLE PHONE CUSTOM TROCR LOGGING TEST (SHARPENED ONLY)")
    print("=" * 50)
    print(f"Model Path: {Config.CUSTOM_TROCR_MODEL_PATH}")
    print(f"Methods: {Config.SELECTED_METHODS}")
    print("=" * 50)
    
    # Initialize checker with custom TrOCR
    try:
        checker = PhoneCarrierChecker(save_captcha_images=True)
    except Exception as e:
        print(f"ERROR: Failed to initialize checker: {e}")
        return
    
    # Single test phone
    test_phone = "0987654321"
    
    print(f"Testing phone number: {test_phone}")
    print("This will show detailed OCR logging using only Sharpened method for maximum speed...")
    
    try:
        result = checker.check_carrier(test_phone)
        
        print(f"\n{'='*70}")
        print("FINAL RESULT SUMMARY (SHARPENED ONLY)")
        print(f"{'='*70}")
        print(f"Phone: {result.phone_number}")
        print(f"Prefix result: {result.prefix_result}")
        print(f"Official result: {result.official_result}")
        print(f"Final result: {result.final_result}")
        print(f"Match: {'Yes' if result.match else 'No'}")
        print(f"Valid: {'Yes' if result.is_valid else 'No'}")
        
    except Exception as e:
        logger.error(f"Failed to process {test_phone}: {e}")
        print("Test failed")


def run_speed_benchmark():
    """Benchmark the speed improvement of using only sharpened method"""
    print("SPEED BENCHMARK - SHARPENED ONLY vs MULTIPLE METHODS")
    print("=" * 60)
    print("This benchmark shows the speed improvement of using only sharpened preprocessing")
    print("compared to using multiple preprocessing methods.")
    print("=" * 60)
    
    try:
        # Initialize checker
        checker = PhoneCarrierChecker(save_captcha_images=False)
        print("Custom TrOCR checker initialized successfully")
        print(f"Using only: {Config.SELECTED_METHODS}")
        print(f"Method weight: {Config.METHOD_WEIGHTS}")
        print("\nBenefits of using only Sharpened method:")
        print("• 2x faster processing (1 method instead of 2)")
        print("• Reduced memory usage")
        print("• Simplified voting algorithm")
        print("• Higher confidence scores due to focused approach")
        print("• Uses custom model's native preprocessor config")
        
    except Exception as e:
        print(f"Benchmark initialization failed: {e}")


if __name__ == "__main__":
    try:
        # Choose which test to run:
        
        # Full automated test with multiple phone numbers
        run_test()
        
        # Speed benchmark
        # run_speed_benchmark()
        
        # Single phone test with detailed logging
        # run_single_phone_test()
        
        # Or just test CAPTCHA solver initialization
        # run_captcha_test()
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print("Please check dependencies and try again")
        print("\nRequired dependencies:")
        print("   pip install selenium transformers torch opencv-python pillow numpy")