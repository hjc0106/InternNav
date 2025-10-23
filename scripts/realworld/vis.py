import glob
from PIL import Image
import matplotlib.pyplot as plt

save_dir = "test_data"
for img_path in sorted(glob.glob(f'{save_dir}/*_annotated.png')):
    plt.imshow(Image.open(img_path))
    plt.axis('off')
    plt.show()