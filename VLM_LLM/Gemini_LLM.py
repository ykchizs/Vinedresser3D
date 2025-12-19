
def identify_ori_part(client, prompt_ori, editing_part):

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=f"This is the description of a 3D object: {prompt_ori} And these are some target parts of it: {editing_part}. Please extract the descriptions of the target parts out. Note that you should only extract the main descriptions of the target parts (output in the format of shape + color + texture + object's name), not their relationships with other parts. The output should be a phrase or a sentence (e.g. a red flat tiled roof).",
    )

    return response.text

def identify_new_part(client, prompt_ori, editing_part):

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=f"This is the description of a 3D object: {prompt_ori} And these are some target parts of it: {editing_part}. Please extract the descriptions of the target parts out. Note that you should only extract the main descriptions of the target parts (output in the format of shape + color + texture + object's name), not their relationships with other parts. The output should be a phrase or a sentence (e.g. a red flat tiled roof).",
    )

    return response.text

def decompose_prompt(client, prompt):

    input_text = f"""   This is the description of a 3D object: {prompt}
    You need to modify the description with the following rules:
    1. Identify all the adjectives.
    2. Remove all the adjectives about the object's detailed appearance (e.g. color, texture, material).
    Output only the modified description.
    """
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=input_text,
    )
    s1 = response.text

    input_text = f"""   This is the description of a 3D object: {prompt}
    You need to modify the description with the following rules:
    1. Identify all the adjectives.
    2. Remove all the adjectives about the object's structure (e.g. shape).
    Output only the modified description.
    """
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=input_text,
    )
    s2 = response.text
    
    return s1, s2
