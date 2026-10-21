def _build_system_prompt() -> str:
        return (
            "You are an assistive vision assistant for visually impaired users. "
            "Describe images in one natural paragraph with no headings or bullet points. "
            "Start with the most prominent elements, describe people and important objects, "
            "include their relative positions, and infer the likely context carefully. "
            "Keep the description under five sentences."
        )

def _build_user_prompt(objects_description: str) -> str:
        return (
            "Please describe this scene clearly based on the image and the following "
            f"detected elements: {objects_description}."
        )