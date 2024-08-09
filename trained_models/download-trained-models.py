#!/usr/bin/env python3

# TODO - Add direct download links

models = [
    ( 'AlexNet_epoch_16_91.2_calibrated.pth', 'https://drive.google.com/file/d/1uCDKoIx7woGinxInA4xGyMWvcwMpKnA7/view?usp=drive_link' ),
    ( 'AlexNet_epoch_16_91.2.pth', 'https://drive.google.com/file/d/15r9b1HlWU_v6qexesVxTjVnzrDZlmpH8/view?usp=drive_link' ),
    ( 'AlexNetWithExits_epoch_19_90.1_91.1_calibrated.pth', 'https://drive.google.com/file/d/18hXOmjgmYNbYNIrKVVAT9SB9WNdVFUfU/view?usp=drive_link' ),
    ( 'AlexNetWithExits_epoch_19_90.1_91.1.pth', 'https://drive.google.com/file/d/1ckwMqDlcLWpSqlaUi1eR7nmCY8bEY6L4/view?usp=drive_link' ),
    ( 'MobileNetV2_epoch_17_90.9.pth', 'https://drive.google.com/file/d/1GCb85UcBNhENtjvXtpYYPNf-f08NLXiN/view?usp=drive_link' ),
    ( 'MobileNetV2WithExits_epoch_19_89.7_90.9_calibrated.pth', 'https://drive.google.com/file/d/1thR1A3tI_9jKAhtCjAUY1fymvHx19SuD/view?usp=drive_link' ),
    ( 'MobileNetV2WithExits_epoch_19_89.7_90.9.pth', 'https://drive.google.com/file/d/1eBhCJSr7GWEVlIAHvCFw3j11Vyv3Y38X/view?usp=drive_link' ),
]

for filename, url in models:
    print(f'Access {url} to download {filename}')