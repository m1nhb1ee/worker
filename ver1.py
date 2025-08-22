class PhoneCarrierChecker:
    def __init__(self):
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

    def normalize(self, phone_number: str) -> str:
        """Chuẩn hóa số điện thoại về dạng bắt đầu bằng 0"""
        phone_number = phone_number.strip()
        if phone_number.startswith("+84"):
            phone_number = "0" + phone_number[3:]
        return phone_number

    def get_carrier(self, phone_number: str) -> str:
        """Xác định nhà mạng của số điện thoại"""
        phone_number = self.normalize(phone_number)
        for carrier, prefixes in self.PREFIX_MAP.items():
            if any(phone_number.startswith(prefix) for prefix in prefixes):
                return carrier
        return "Không xác định"

    def check_list(self, phone_list):
        """Trả về dict chứa số và nhà mạng"""
        return {num: self.get_carrier(num) for num in phone_list}


# --- Demo ---
if __name__ == "__main__":
    checker = PhoneCarrierChecker()
    phone_numbers = [
        "0987654321",
        "+84912345678",
        "0761234567",
        "0558888888",
        "+84123123123"
    ]
    results = checker.check_list(phone_numbers)
    for num, carrier in results.items():
        print(f"{num} → {carrier}")
