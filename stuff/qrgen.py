import qrcode
from qrcode.constants import ERROR_CORRECT_H

# URL to your website
url = "https://www.trivecbuilders.com"

# Generate the QR code
qr = qrcode.QRCode(
    version=1,
    error_correction=ERROR_CORRECT_H,
    box_size=10,
    border=4,
)
qr.add_data(url)
qr.make(fit=True)

# Create an image
img = qr.make_image(fill_color="black", back_color="white")

# Save it
with open("trivecbuildersqr.png", "wb") as f:
    img.save(f)

print("✅ QR code saved as trivecbuildersqr.png")
