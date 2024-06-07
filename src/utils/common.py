import PIL
import io

def figure2pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches="tight", pad_inches=0.1)
    buf.seek(0)
    return PIL.Image.open(buf)

