import matplotlib.pyplot as plt


def show_img(img):
    plt.figure(figsize=(20, 4))
    plt.imshow(img)
    plt.show()

def show_img_arr(img_arr, n_image_per_row):
    n_image = len(img_arr)
    plt.figure(figsize=(20, 4))
    for idx, img in enumerate(img_arr):
        ax = plt.subplot(int(n_image/n_image_per_row+1), n_image_per_row, idx + 1)
        plt.imshow(img)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
