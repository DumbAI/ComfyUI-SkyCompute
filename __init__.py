"""ComfyUI-SkyCompute: nodes for the Sky Compute platform's services.

ComfyUI loads this package from custom_nodes/ and calls `comfy_entrypoint`
(the V3 node API).
"""

# pytest imports this file as a plain module (the repository root has no parent
# package); the tests import sky_compute directly and need no ComfyUI.
if __package__:
    from .sky_compute.nodes import SkyComputeExtension

    async def comfy_entrypoint() -> SkyComputeExtension:
        return SkyComputeExtension()
