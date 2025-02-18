def cot_fewshot_using_categories(prompt, example_pre_post, example_training, example_classification, example_image):
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
    
