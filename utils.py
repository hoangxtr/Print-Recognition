import os

def ocr(fname):
    if fname == 'image1.tif':
        ocr_ret = 'pro 2022/05/13\nexp=2023/05/12'
    elif fname in ['image2.tif', 'image3.tif', 'image5.tif', ]:
        ocr_ret = 'best before 20/11/2023'
    elif fname == 'image4.tif':
        ocr_ret = 'lot 123456\nbb/ma 2023 no 11'
    elif fname == 'image6.tif':
        ocr_ret = 'nsx 09/05/2022\nhsd 08/05/2023'

    return ocr_ret
