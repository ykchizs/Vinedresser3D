
from PIL import Image
from io import BytesIO

def Nano_banana_edit(client, img_path, edit_prompt, new_part=None):

    image_input = Image.open(img_path)
    text_input = f"""
    Edit the image provided with the editing prompt: {edit_prompt}
    and the guidance of the new parts: 
    {new_part}"""

    # Generate an image from a text prompt
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[text_input, image_input],
    )

    image_parts = [
        part.inline_data.data
        for part in response.candidates[0].content.parts
        if part.inline_data
    ]

    if image_parts:
        image = Image.open(BytesIO(image_parts[0]))
    else:
        raise ValueError("No image parts found")
    
    return image