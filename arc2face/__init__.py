from .models import CLIPTextModelWrapper
from .utils import project_face_embs, image_align
from .loader import (
    encode_input_image,
    generate_from_embedding,
    load_arc2face_pipeline,
    resolve_local_snapshot,
)

# ReferenceAdapter is imported lazily because it requires `einops`, which is not
# pulled in by the core analysis path. Use `from arc2face.ref_adapter import ReferenceAdapter`
# directly when you need it.