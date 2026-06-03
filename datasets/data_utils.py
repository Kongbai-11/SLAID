import numpy as np
import torch 
import cv2


def mask_score(mask):
    '''Scoring the mask according to connectivity.'''
    mask = mask.astype(np.uint8)
    if mask.sum() < 10:
        return 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt_area = [cv2.contourArea(cnt) for cnt in contours]
    conc_score = np.max(cnt_area) / sum(cnt_area)
    return conc_score


def _estimate_mask_angle(mask):
    pts = np.column_stack(np.where(mask > 0))
    if len(pts) < 2:
        return 0.0
    ys = pts[:, 0].astype(np.float32)
    xs = pts[:, 1].astype(np.float32)
    coords = np.stack([xs, ys], axis=1)
    coords -= coords.mean(axis=0, keepdims=True)
    cov = np.cov(coords, rowvar=False)
    if np.any(~np.isfinite(cov)):
        return 0.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, np.argmax(eigvals)]
    return float(np.arctan2(v[1], v[0]))


def _major_is_horizontal(angle):
    return abs(math.cos(angle)) >= abs(math.sin(angle))


def _morph_skeleton(binary_mask):
    img = (binary_mask > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return (skel > 0).astype(np.uint8)


def _fft_periodic_residual(gray_float, top_k=8, center_radius=7, notch_radius=3, attenuate=0.10):
    h, w = gray_float.shape
    F = np.fft.fftshift(np.fft.fft2(gray_float))
    mag = np.log1p(np.abs(F))
    cy, cx = h // 2, w // 2

    yy, xx = np.ogrid[:h, :w]
    central = (yy - cy) ** 2 + (xx - cx) ** 2 <= center_radius ** 2
    mag_work = mag.copy()
    mag_work[central] = 0.0

    notch_mask = np.ones((h, w), dtype=np.float32)
    flat_idx = np.argpartition(mag_work.ravel(), -min(top_k, mag_work.size))[-min(top_k, mag_work.size):]
    pts = np.column_stack(np.unravel_index(flat_idx, mag_work.shape))

    for py, px in pts:
        if (py - cy) ** 2 + (px - cx) ** 2 <= center_radius ** 2:
            continue
        sy, sx = 2 * cy - py, 2 * cx - px
        cv2.circle(notch_mask, (int(px), int(py)), int(notch_radius), float(attenuate), -1)
        if 0 <= sy < h and 0 <= sx < w:
            cv2.circle(notch_mask, (int(sx), int(sy)), int(notch_radius), float(attenuate), -1)

    F_filtered = F * notch_mask
    recon = np.real(np.fft.ifft2(np.fft.ifftshift(F_filtered))).astype(np.float32)
    residual = np.abs(gray_float - recon)
    return residual


def _normalize_in_support(x, support, p_lo=40.0, p_hi=99.5):
    vals = x[support > 0]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo = float(np.percentile(vals, p_lo))
    hi = float(np.percentile(vals, p_hi))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def dog(img, mask, thresh=18, top_k=8, notch_radius=3, keep_percent=45.0):
    """
    FFT + mask-geometry high-frequency map.
    Output style is close to official AnyDoor high-frequency guidance:
    black background + sparse white/gray lines in 3-channel uint8.
    """
    H, W = img.shape[:2]
    img_256 = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
    mask_256 = (cv2.resize(mask.astype(np.float32), (256, 256), interpolation=cv2.INTER_NEAREST) > 0.5).astype(np.uint8)

    if mask_256.sum() == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    mask_erode = cv2.erode(mask_256, np.ones((5, 5), np.uint8), iterations=1)
    if mask_erode.sum() < 16:
        mask_erode = mask_256.copy()

    angle = _estimate_mask_angle(mask_erode)
    vertical = not _major_is_horizontal(angle)
    band_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15) if vertical else (15, 5))
    support = cv2.dilate(mask_erode, band_kernel, iterations=1)

    gray = cv2.cvtColor(img_256, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    residual = _fft_periodic_residual(gray, top_k=top_k, center_radius=7, notch_radius=notch_radius, attenuate=0.10)
    residual_n = _normalize_in_support(residual, support, p_lo=30.0, p_hi=99.7)

    gradx = cv2.Sobel((gray * 255).astype(np.uint8), cv2.CV_32F, 1, 0, ksize=3)
    grady = cv2.Sobel((gray * 255).astype(np.uint8), cv2.CV_32F, 0, 1, ksize=3)
    grad = 0.5 * np.abs(gradx) + 0.5 * np.abs(grady)
    grad_n = _normalize_in_support(grad.astype(np.float32), support, p_lo=55.0, p_hi=99.5)

    edge = cv2.morphologyEx(mask_erode, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(np.float32)
    edge = np.clip(edge, 0.0, 1.0)

    center = _morph_skeleton(mask_erode).astype(np.float32)
    if center.sum() == 0:
        center = mask_erode.astype(np.float32)
    center = cv2.dilate(center, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    center = np.clip(center, 0.0, 1.0)

    resp = 0.42 * residual_n + 0.12 * grad_n + 0.24 * edge + 0.22 * center
    resp *= support.astype(np.float32)

    inside_vals = resp[mask_erode > 0]
    if inside_vals.size == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    keep_thr = float(np.percentile(inside_vals, max(0.0, 100.0 - float(keep_percent))))
    base_thr = max(thresh / 255.0, keep_thr)
    binary = (resp >= base_thr).astype(np.uint8)
    binary = np.maximum(binary, center.astype(np.uint8))

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7) if vertical else (7, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    binary = (binary * support).astype(np.uint8)

    intensity = np.zeros_like(resp, dtype=np.float32)
    intensity += binary.astype(np.float32) * 0.72
    intensity += np.clip(resp, 0.0, 1.0) * 0.28
    intensity = np.clip(intensity, 0.0, 1.0)
    intensity = cv2.GaussianBlur(intensity, (3, 3), 0)
    intensity = np.clip(intensity * 255.0, 0, 255).astype(np.uint8)

    hf = np.stack([intensity, intensity, intensity], axis=-1)
    hf = cv2.resize(hf, (W, H), interpolation=cv2.INTER_LINEAR)
    return hf



def sobel(img, mask, thresh = 50):
    '''Calculating the high-frequency map.'''
    H,W = img.shape[0], img.shape[1]
    img = cv2.resize(img,(256,256))
    mask = (cv2.resize(mask,(256,256)) > 0.5).astype(np.uint8)
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.erode(mask, kernel, iterations = 2)
    
    Ksize = 3
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=Ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=Ksize)
    sobel_X = cv2.convertScaleAbs(sobelx)
    sobel_Y = cv2.convertScaleAbs(sobely)
    scharr = cv2.addWeighted(sobel_X, 0.5, sobel_Y, 0.5, 0)
    scharr = np.max(scharr,-1) * mask    
    
    scharr[scharr < thresh] = 0.0
    scharr = np.stack([scharr,scharr,scharr],-1)
    scharr = (scharr.astype(np.float32)/255 * img.astype(np.float32) ).astype(np.uint8)
    scharr = cv2.resize(scharr,(W,H))
    return scharr


def resize_and_pad(image, box):
    '''Fitting an image to the box region while keeping the aspect ratio.'''
    y1,y2,x1,x2 = box
    H,W = y2-y1, x2-x1
    h,w =  image.shape[0], image.shape[1]
    r_box = W / H 
    r_image = w / h
    if r_box >= r_image:
        h_target = H
        w_target = int(w * H / h) 
        image = cv2.resize(image, (w_target, h_target))

        w1 = (W - w_target) // 2
        w2 = W - w_target - w1
        pad_param = ((0,0),(w1,w2),(0,0))
        image = np.pad(image, pad_param, 'constant', constant_values=255)
    else:
        w_target = W 
        h_target = int(h * W / w)
        image = cv2.resize(image, (w_target, h_target))

        h1 = (H-h_target) // 2 
        h2 = H - h_target - h1
        pad_param =((h1,h2),(0,0),(0,0))
        image = np.pad(image, pad_param, 'constant', constant_values=255)
    return image



def expand_image_mask(image, mask, ratio=1.4):
    h,w = image.shape[0], image.shape[1]
    H,W = int(h * ratio), int(w * ratio) 
    h1 = int((H - h) // 2)
    h2 = H - h - h1
    w1 = int((W -w) // 2)
    w2 = W -w - w1

    pad_param_image = ((h1,h2),(w1,w2),(0,0))
    pad_param_mask = ((h1,h2),(w1,w2))
    image = np.pad(image, pad_param_image, 'constant', constant_values=255)
    mask = np.pad(mask, pad_param_mask, 'constant', constant_values=0)
    return image, mask


def resize_box(yyxx, H,W,h,w):
    y1,y2,x1,x2 = yyxx
    y1,y2 = int(y1/H * h), int(y2/H * h)
    x1,x2 = int(x1/W * w), int(x2/W * w)
    y1,y2 = min(y1,h), min(y2,h)
    x1,x2 = min(x1,w), min(x2,w)
    return (y1,y2,x1,x2)


def get_bbox_from_mask(mask):
    h,w = mask.shape[0],mask.shape[1]

    if mask.sum() < 10:
        return 0,h,0,w
    rows = np.any(mask,axis=1)
    cols = np.any(mask,axis=0)
    y1,y2 = np.where(rows)[0][[0,-1]]
    x1,x2 = np.where(cols)[0][[0,-1]]
    return (y1,y2,x1,x2)


def expand_bbox(mask,yyxx,ratio=[1.2,2.0], min_crop=0):
    y1,y2,x1,x2 = yyxx
    ratio = np.random.randint( ratio[0] * 10,  ratio[1] * 10 ) / 10
    H,W = mask.shape[0], mask.shape[1]
    xc, yc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    h = ratio * (y2-y1+1)
    w = ratio * (x2-x1+1)
    h = max(h,min_crop)
    w = max(w,min_crop)

    x1 = int(xc - w * 0.5)
    x2 = int(xc + w * 0.5)
    y1 = int(yc - h * 0.5)
    y2 = int(yc + h * 0.5)

    x1 = max(0,x1)
    x2 = min(W,x2)
    y1 = max(0,y1)
    y2 = min(H,y2)
    return (y1,y2,x1,x2)


def box2squre(image, box):
    H,W = image.shape[0], image.shape[1]
    y1,y2,x1,x2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    h,w = y2-y1, x2-x1

    if h >= w:
        x1 = cx - h//2
        x2 = cx + h//2
    else:
        y1 = cy - w//2
        y2 = cy + w//2
    x1 = max(0,x1)
    x2 = min(W,x2)
    y1 = max(0,y1)
    y2 = min(H,y2)
    return (y1,y2,x1,x2)


def pad_to_square(image, pad_value = 255, random = False):
    H,W = image.shape[0], image.shape[1]
    if H == W:
        return image

    padd = abs(H - W)
    if random:
        padd_1 = int(np.random.randint(0,padd))
    else:
        padd_1 = int(padd / 2)
    padd_2 = padd - padd_1

    if H > W:
        pad_param = ((0,0),(padd_1,padd_2),(0,0))
    else:
        pad_param = ((padd_1,padd_2),(0,0),(0,0))

    image = np.pad(image, pad_param, 'constant', constant_values=pad_value)
    return image



def box_in_box(small_box, big_box):
    y1,y2,x1,x2 = small_box
    y1_b, _, x1_b, _ = big_box
    y1,y2,x1,x2 = y1 - y1_b ,y2 - y1_b, x1 - x1_b ,x2 - x1_b
    return (y1,y2,x1,x2 )



def shuffle_image(image, N):
    height, width = image.shape[:2]
    
    block_height = height // N
    block_width = width // N
    blocks = []
    
    for i in range(N):
        for j in range(N):
            block = image[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width]
            blocks.append(block)
    
    np.random.shuffle(blocks)
    shuffled_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(N):
        for j in range(N):
            shuffled_image[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width] = blocks[i*N+j]
    return shuffled_image


def get_mosaic_mask(image, fg_mask, N=16, ratio = 0.5):
    ids = [i for i in range(N * N)]
    masked_number = int(N * N * ratio)
    masked_id = np.random.choice(ids, masked_number, replace=False)
    

    
    height, width = image.shape[:2]
    mask = np.ones((height, width))
    
    block_height = height // N
    block_width = width // N
    
    b_id = 0
    for i in range(N):
        for j in range(N):
            if b_id in masked_id:
                mask[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width] = mask[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width] * 0
            b_id += 1
    mask = mask * fg_mask
    mask3 = np.stack([mask,mask,mask],-1).copy().astype(np.uint8)
    noise = q_x(image)
    noise_mask = image * mask3 + noise * (1-mask3)
    return noise_mask

def extract_canney_noise(image, mask, dilate=True):
    h,w = image.shape[0],image.shape[1]
    mask = cv2.resize(mask.astype(np.uint8),(w,h)) > 0.5
    kernel = np.ones((8, 8), dtype=np.uint8)
    mask =  cv2.erode(mask.astype(np.uint8), kernel, 10)

    canny = cv2.Canny(image, 50,100) * mask
    kernel = np.ones((8, 8), dtype=np.uint8)
    mask = (cv2.dilate(canny, kernel, 5) > 128).astype(np.uint8)
    mask = np.stack([mask,mask,mask],-1)

    pure_noise = q_x(image, t=1) * 0 + 255
    canny_noise = mask * image + (1-mask) * pure_noise
    return canny_noise


def get_random_structure(size):
    choice = np.random.randint(1, 5)

    if choice == 1:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif choice == 2:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif choice == 3:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size//2))
    elif choice == 4:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size//2, size))

def random_dilate(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.dilate(seg,kernel,iterations = 1)
    return seg

def random_erode(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.erode(seg,kernel,iterations = 1)
    return seg

def compute_iou(seg, gt):
    intersection = seg*gt
    union = seg+gt
    return (np.count_nonzero(intersection) + 1e-6) / (np.count_nonzero(union) + 1e-6)


def select_max_region(mask):
    nums, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    background = 0
    for row in range(stats.shape[0]):
        if stats[row, :][0] == 0 and stats[row, :][1] == 0:
            background = row
    stats_no_bg = np.delete(stats, background, axis=0)
    max_idx = stats_no_bg[:, 4].argmax()
    max_region = np.where(labels==max_idx+1, 1, 0)

    return max_region.astype(np.uint8)



def perturb_mask(gt, min_iou = 0.3,  max_iou = 0.99):
    iou_target = np.random.uniform(min_iou, max_iou)
    h, w = gt.shape
    gt = gt.astype(np.uint8)
    seg = gt.copy()
    
    # Rare case
    if h <= 2 or w <= 2:
        print('GT too small, returning original')
        return seg

    # Do a bunch of random operations
    for _ in range(250):
        for _ in range(4):
            lx, ly = np.random.randint(w), np.random.randint(h)
            lw, lh = np.random.randint(lx+1,w+1), np.random.randint(ly+1,h+1)

            # Randomly set one pixel to 1/0. With the following dilate/erode, we can create holes/external regions
            if np.random.rand() < 0.1:
                cx = int((lx + lw) / 2)
                cy = int((ly + lh) / 2)
                seg[cy, cx] = np.random.randint(2) * 255

            # Dilate/erode
            if np.random.rand() < 0.5:
                seg[ly:lh, lx:lw] = random_dilate(seg[ly:lh, lx:lw])
            else:
                seg[ly:lh, lx:lw] = random_erode(seg[ly:lh, lx:lw])
            
            seg = np.logical_or(seg, gt).astype(np.uint8)
            #seg = select_max_region(seg) 

        if compute_iou(seg, gt) < iou_target:
            break
    seg = select_max_region(seg.astype(np.uint8)) 
    return seg.astype(np.uint8)


def q_x(x_0,t=65):
    '''Adding noise for and given image.'''
    x_0 = torch.from_numpy(x_0).float() / 127.5 - 1
    num_steps = 100
    
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas)*(0.5e-2 - 1e-5)+1e-5

    alphas = 1-betas
    alphas_prod = torch.cumprod(alphas,0)
    
    alphas_prod_p = torch.cat([torch.tensor([1]).float(),alphas_prod[:-1]],0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
    
    noise = torch.randn_like(x_0)
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    return (alphas_t * x_0 + alphas_1_m_t * noise).numpy()  * 127.5 + 127.5 


def extract_target_boundary(img, target_mask):
    Ksize = 3
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=Ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=Ksize)

    # sobel-x
    sobel_X = cv2.convertScaleAbs(sobelx)
    # sobel-y
    sobel_Y = cv2.convertScaleAbs(sobely)
    # sobel-xy
    scharr = cv2.addWeighted(sobel_X, 0.5, sobel_Y, 0.5, 0)
    scharr = np.max(scharr,-1).astype(np.float32)/255
    scharr = scharr *  target_mask.astype(np.float32)
    return scharr