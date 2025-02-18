def cot_fewshot_data_outsider(prompt):
    return f'''
    Here are some examples of how to generate the code step by step.

    Example 1:

    ```python
    def draw_bounding_boxes(
        image: torch.Tensor,
        boxes: torch.Tensor,
        labels: Optional[List[str]] = None,
        colors: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]] = None,
        fill: Optional[bool] = False,
        width: int = 1,
        font: Optional[str] = None,
        font_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Draws bounding boxes on a given RGB image.

        The image values should be uint8 in [0, 255] or float in [0, 1].
        If fill is True, the resulting Tensor should be saved as a PNG image to preserve the fill.

        Args:
            image (Tensor): Tensor of shape (C, H, W) and dtype uint8 or float.
            boxes (Tensor): Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax) format.
            labels (List[str], optional): Labels for the bounding boxes. Defaults to None.
            colors (Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]], optional):
                Colors for each bounding box or a single color for all boxes.
            fill (bool): Whether to fill the bounding boxes.
            width (int): Width of the bounding box outline.
            font (str, optional): Path to a TrueType font. Defaults to None.
            font_size (int, optional): Font size for labels. Defaults to None.

        Returns:
            img (Tensor[C, H, W]): Image tensor (dtype=uint8) with bounding boxes drawn.
        """
        import torchvision.transforms.v2.functional as F  # noqa

        # ---------------------------------------------------------------
        # 1) Determine how many bounding boxes we have
        # ---------------------------------------------------------------
        num_boxes = boxes.shape[0]

        # ---------------------------------------------------------------
        # 2) If there are no boxes, warn and return the original image
        # ---------------------------------------------------------------
        if num_boxes == 0:
            warnings.warn("boxes doesn't contain any box. No box was drawn")
            return image

        # ---------------------------------------------------------------
        # 3) Prepare labels. If none are provided, create placeholders.
        #    Ensure the label list length matches the number of boxes.
        # ---------------------------------------------------------------
        if labels is None:
            labels: Union[List[str], List[None]] = [None] * num_boxes
        elif len(labels) != num_boxes:
            raise ValueError(
                f"Number of boxes (num_boxes) and labels (len(labels)) mismatch."
                " Please specify labels for each box."
            )

        # ---------------------------------------------------------------
        # 4) Parse the colors provided. If a single color is given,
        #    use it for all boxes; otherwise match colors to boxes.
        # ---------------------------------------------------------------
        colors = _parse_colors(colors, num_objects=num_boxes)

        # ---------------------------------------------------------------
        # 5) Load or default to a basic font. If 'font' is not set,
        #    ignore the 'font_size' argument unless explicitly used later.
        # ---------------------------------------------------------------
        if font is None:
            if font_size is not None:
                warnings.warn("Argument 'font_size' will be ignored since 'font' is not set.")
            txt_font = ImageFont.load_default()
        else:
            txt_font = ImageFont.truetype(font=font, size=font_size or 10)

        # ---------------------------------------------------------------
        # 6) Handle grayscale images (C=1) by converting them to
        #    three-channel images for consistent bounding box drawing.
        # ---------------------------------------------------------------
        if image.size(0) == 1:
            image = torch.tile(image, (3, 1, 1))

        # ---------------------------------------------------------------
        # 7) Convert image to uint8 if it’s floating-point,
        #    because drawing operations typically require uint8.
        # ---------------------------------------------------------------
        original_dtype = image.dtype
        if original_dtype.is_floating_point:
            image = F.to_dtype(image, dtype=torch.uint8, scale=True)

        # ---------------------------------------------------------------
        # 8) Convert the PyTorch tensor to a PIL Image for using
        #    PIL’s drawing functions.
        # ---------------------------------------------------------------
        img_to_draw = F.to_pil_image(image)

        # ---------------------------------------------------------------
        # 9) Convert bounding boxes to int and then to Python lists
        #    because PIL draw methods need standard Python data types.
        # ---------------------------------------------------------------
        img_boxes = boxes.to(torch.int64).tolist()

        # ---------------------------------------------------------------
        # 10) Create a draw object. If fill is True, use "RGBA" mode
        #     for partial transparency (fill color).
        # ---------------------------------------------------------------
        if fill:
            draw = ImageDraw.Draw(img_to_draw, "RGBA")
        else:
            draw = ImageDraw.Draw(img_to_draw)

        # ---------------------------------------------------------------
        # 11) Loop over each bounding box, color, and label.
        #     Draw rectangle outlines; optionally fill and add text.
        # ---------------------------------------------------------------
        for bbox, color, label in zip(img_boxes, colors, labels):
            if fill:
                # Semi-transparent fill
                fill_color = color + (100,)
                draw.rectangle(bbox, width=width, outline=color, fill=fill_color)
            else:
                draw.rectangle(bbox, width=width, outline=color)

            # -----------------------------------------------------------
            # 12) If a label is provided, place it near the top-left
            #     corner of the bounding box.
            # -----------------------------------------------------------
            if label is not None:
                margin = width + 1
                draw.text((bbox[0] + margin, bbox[1] + margin), label, fill=color, font=txt_font)

        # ---------------------------------------------------------------
        # 13) Convert the PIL image back to a tensor. Restore original
        #     floating-point type if needed.
        # ---------------------------------------------------------------
        out = F.pil_to_tensor(img_to_draw)
        if original_dtype.is_floating_point:
            out = F.to_dtype(out, dtype=original_dtype, scale=True)

        # ---------------------------------------------------------------
        # 14) Return the resulting tensor with bounding boxes.
        # ---------------------------------------------------------------
        return out

    ```

    Example 2:

    ```python
    import numpy as np
    from sklearn.utils import check_random_state

    def make_hastie_10_2(n_samples=12000, *, random_state=None):
        """
        Generate a binary classification dataset (Hastie et al., Example 10.2).
        
        The data has 10 Gaussian features, and the target is 1 if the sum 
        of squares of features > 9.34, otherwise -1.

        """
        # 1. Convert random_state into a RandomState instance
        rs = check_random_state(random_state)
        
        # 2. Define the shape of the dataset: (n_samples, 10 features)
        shape = (n_samples, 10)
        
        # 3. Generate random Gaussian data of the specified shape
        X = rs.normal(size=shape).reshape(shape)
        
        # 4. Compute the sum of squares of each sample's features
        #    and check if it's greater than 9.34
        y = ((X ** 2).sum(axis=1) > 9.34).astype(np.float64, copy=False)
        
        # 5. Replace all 0.0 labels with -1.0 to create a binary output (-1, 1)
        y[y == 0.0] = -1.0
        
        # 6. Return the generated features X and corresponding labels y
        return X, y

    ```

    Example 3:

    ```python
   import numpy as np

def frequencies_to_period(frequencies, decimals=5):
    """
    Calculate the period of frequencies as 2π / gcd(frequencies).
    
    If the frequencies are non-integral, they are rounded to `decimals` places 
    before computing the gcd.

    """
    # 1. Try to directly compute gcd for the frequencies (works if they're integers).
    try:
        gcd = np.gcd.reduce(frequencies)
    except TypeError:
        # 2. If we have non-integer frequencies, we:
        #    - Round them to `decimals` places
        #    - Scale them by 10^decimals to make them integers
        exponent = 10 ** decimals
        rounded = np.round(frequencies, decimals) * exponent
        
        # 3. Convert to int and compute gcd, then scale gcd back down
        gcd = np.gcd.reduce(np.int64(rounded)) / exponent

    # 4. Return 2π divided by the gcd for the final period
    return 2 * np.pi / gcd

    ```
    
    Example 4: 
    ```python
    import numpy as np
    from itertools import product

    def _coefficients_no_filter(f, degree, use_broadcasting):
        """
        Compute the raw Fourier coefficients (2d+1 terms) of a 2π-periodic function.
        # 1. Convert degree to a NumPy array for consistent handling
        degree = np.array(degree)

        # 2. Determine how many frequencies to include:
        #    For each dimension d_i, we have frequencies -d_i ... 0 ... d_i,
        #    which totals 2*d_i + 1 values per dimension.
        k = 2 * degree + 1  # shape in each dimension

        # 3. Create a list of ranges for each dimension: [-d_i, ..., d_i]
        n_ranges = [np.arange(-d, d + 1) for d in degree]

        # 4. Prepare to iterate over all combinations (Cartesian product) of the n_ranges
        #    except possibly the last one if broadcasting is used.
        nvecs = product(*(n_ranges[:-1] if use_broadcasting else n_ranges))

        # 5. Initialize a NumPy array to hold the discretized function values.
        #    The shape is determined by k for each dimension.
        f_discrete = np.zeros(shape=tuple(k), dtype=complex)

        # 6. Compute the spacing for each dimension, used to map indices -> evaluation points
        spacing = (2 * np.pi) / k

        # 7. Loop over all combinations of indices (nvec) to fill f_discrete
        for nvec in nvecs:
            # Handle broadcasting:
            # if use_broadcasting, append the entire last dimension's n-range as an array.
            if use_broadcasting:
                nvec = (*nvec, n_ranges[-1])  # n_ranges[-1] is an array
                sampling_point = [s * n for s, n in zip(spacing, nvec)]
            else:
                sampling_point = spacing * np.array(nvec)

            # Evaluate f at the computed sampling_point and store
            f_discrete[nvec] = f(sampling_point)

        # 8. Use multi-dimensional FFT to compute Fourier coefficients
        #    Divide by the total number of points to normalize
        coeffs = np.fft.fftn(f_discrete) / f_discrete.size

        # 9. Return the raw Fourier coefficients
        return coeffs

    
    ```

    How about this function?
    {prompt}
    '''
    
def cot_fewshot_using_different_categories(prompt, example_pre_post, example_training, example_classification, example_image):
    return f'''
    Here are some examples of how to generate the code for deep learning step by step.

    Example 1:
    Here is an example of pre-post processing stage
    {example_pre_post}

    Example 2:
    Here is an example of training stage
    {example_training}

    Example 3:
    Here is an example for classification task
    {example_classification}
    ```
    
    Example 4: 
    Here is an example for image data
    {example_image}
    ```

    How about this function?
    {prompt}
    '''
    

def cot_few_shot_using_same_or_different_category(prompt, example1, example2, example3, example4, category, is_same = False):
    base_shots = f'''
        Here are some examples of how to generate the code for deep learning step by step.

        #Example 1:
        Here is an example of pre-post processing stage
        {example1}

        Example 2:
        Here is an example of training stage
        {example2}

        Example 3:
        Here is an example for classification task
        {example3}
        ```
        
        Example 4: 
        Here is an example for image data
        {example4}
        ```
        '''
    if not is_same:
        return base_shots + f'''
            How about this function?
            {prompt}
            '''
    else:
        return base_shots + f'''
            How about this function if we assume it is in category of {category}?
            {prompt}
            '''
            

