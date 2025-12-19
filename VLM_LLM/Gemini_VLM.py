
from google.genai import types
import re

def select_K(client, seg_dir, name):

    contents = ["The following images are from different views of the same object. The first 8 images are of the object. The rest are of the object after 3D segmentation. There are 8 images for each K (number of segmentation parts) in order and K ranges from 3 to 8. Please select the best K. Output a single number K."]

    for i in range(8):
        with open(f'outputs/img_multiview/{name}_{i:03d}.png', 'rb') as f:
            img_bytes = f.read()
            contents.append(types.Part.from_bytes(
                data=img_bytes,
                mime_type='image/png'
            ))
    for K in range(3, 9):
        for i in range(8):
            with open(f"{seg_dir}/{name}_{K:02d}_seg_view{i}.png", 'rb') as f:
                img_bytes = f.read()
                contents.append(types.Part.from_bytes(
                    data=img_bytes,
                    mime_type='image/png'
                ))
    
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=contents
    )

    print(response.text)

    # Extract the last number from the response
    for i in range(len(response.text)-1, -1, -1):
        if response.text[i].isdigit():
            return int(response.text[i])

def select_editing_parts(client, seg_dir, name, editing_part):

    contents = [f"""
    The following 8 images are from different views of an object.
    """]
    for i in [0, 1, 6, 7, 4, 5, 2, 3]:
        with open(f"outputs/img_multiview/{name}_{i:03d}.png", 'rb') as f:
            imgs_bytes = f.read()
            contents.append(types.Part.from_bytes(
                data=imgs_bytes,
                mime_type='image/png'
            ))
    
    for k in range(3, 9):
        contents.append(f"""
        The following 8 images are the segmentation result of the object with K={k} parts.
        """)
        for i in range(8):
            with open(f"{seg_dir}/{name}_{k:02d}_seg_view{i}.png", 'rb') as f:
                imgs_bytes = f.read()
                contents.append(types.Part.from_bytes(
                    data=imgs_bytes,
                    mime_type='image/png'
                ))
    
    contents.append(f"""
    I want to identify the parts "{editing_part}" in the object.
    Please give me a K that can achieve this and the colors of the parts that I should select in the segmentation result of that K.
    Note that we only have the colors red, yellow, blue, green, purple, brown, orange, black here.
    Only output the value of K and the colors in the format of value_of_K&&&color1,color2,...
    """)
    
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=contents
    )

    print(response.text)

    return response.text

def select_the_best_edited_object(client, name, editing_prompt):

    contents = [f"""
    The following images are from different views of an object.
    """]
    
    for j in range(8):
        with open(f"outputs/img_multiview/{name}_{j:03d}.png", 'rb') as f:
            imgs_bytes = f.read()
            contents.append(types.Part.from_bytes(
                data=imgs_bytes,
                mime_type='image/png'
            ))
    
    contents.append(f"And this is the editing prompt: {editing_prompt}")
    
    for i in range(5):
        contents.append(f"""
        The following 16 images are from different views of the edited object with index={i}.
        """)
        
        for j in range(16):
            with open(f"outputs/img_multiview/{name}_edited_comb{i}_{j:03d}.png", 'rb') as f:
                imgs_bytes = f.read()
                contents.append(types.Part.from_bytes(
                    data=imgs_bytes,
                    mime_type='image/png'
                ))
    
    contents.append("""Please select the index of the best edited object (starting from 0).""")
    
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=contents
    )

    print(response.text)

    matches = re.findall(r'\d+', response.text)
    if matches:
        best_index = int(matches[-1])
    else:
        best_index = None

    return best_index
    

def obtain_overall_prompts(client, editing_prompt, name):

    imgs_bytes = {}
    for i in range(8):
        with open(f"outputs/img_multiview/{name}_{i:03d}.png", 'rb') as f:
            imgs_bytes[i] = f.read()
    
    input_text = f"""
    The following images are from different views of a 3D object. 
    And this is the editing prompt: {editing_prompt}
    Give me the following texts:
    1. ori_cpl: Describe the object in a single sentence. You should describe all parts of the object and include the information about structure, shape, color and texture of each part.
    2. editing_part: Output only the names of the parts that need to be changed in the original object.
    3. new_cpl: Describe the modified object. Here you should retain all the information of the original object that is irrelevant to the editing prompt.
    4. target_part: Output only the names of the parts in the edited object that are unique or changed.
    5. editing_type: Classify the type of the editing prompt into one of [Addition, Modification, Deletion]. Only output the category without any punctuation.
    Your output should be in the format of ori_cpl&&&editing_part&&&new_cpl&&&target_part&&&editing_type. Only output the text without ori_cpl/editing_part/new_cpl/target_part/editing_type.
    """
    contents = [input_text]
    for i in range(8):
        contents.append(types.Part.from_bytes(
            data=imgs_bytes[i],
            mime_type='image/png'
        ))
    
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=contents
    )

    print(response.text)

    return response.text

def select_img_to_edit(client, name, editing_prompt):

    contents = [f"""
    The following images are from different views of a 3D object.
    And this is the editing prompt: {editing_prompt}
    Please select the image that 
    1. Most clearly shows the part to be edited.
    2. Is the easiest for an image editing model to edit.
    3. Can resonably show the whole object.
    Only output the index of the image (starting from 0).
    """]

    for i in range(24):
        with open(f"outputs/img_multiview/{name}_preEdit{i:03d}.png", 'rb') as f:
            imgs_bytes = f.read()
            contents.append(types.Part.from_bytes(
                data=imgs_bytes,
                mime_type='image/png'
            ))
    
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=contents
    )
    
    print(response.text)

    match = re.findall(r'\d+', response.text)
    if match:
        last_number = int(match[-1])
    else:
        last_number = None

    return last_number