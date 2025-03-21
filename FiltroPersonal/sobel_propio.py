import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt

def conv_helper(fragment, kernel):
    """Multiplica dos matrices y devuelve su suma"""
    return np.sum(fragment * kernel)

def convolution(image, kernel, padding=0):
    """
    Aplica una convolución con opción de padding.
    :param image: Imagen de entrada en escala de grises.
    :param kernel: Matriz de filtro.
    :param padding: Tamaño del padding a aplicar.
    :return: Imagen resultante después de la convolución.
    """
    image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
    
    output = np.zeros((image_row - kernel_row + 1, image_col - kernel_col + 1))
    
    for row in range(output.shape[0]):
        for col in range(output.shape[1]):
            output[row, col] = conv_helper(
                image[row:row + kernel_row, col:col + kernel_col], kernel
            )
    
    return output

def apply_custom_filter(image, filter, padding=0, verbose=False):
    """Aplica un filtro personalizado a la imagen mediante convolución."""
    filtered_image = convolution(image, filter, padding)
    
    if verbose:
        plt.imshow(filtered_image, cmap='gray')
        plt.title("Imagen Filtrada (Nitidez/Contraste)")
        plt.show()
    
    return filtered_image

if __name__ == '__main__':
    # FILTRO (Nitidez/Contraste)
    filter = np.array([[ 1,  1,  1],  
                       [ 0,  3,  0],  
                       [-1, -1, -1]])  

    # Obtiene la imagen desde la línea de comandos "python script.py -i img.jpg"
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path of image")
    ap.add_argument("-p", "--padding", type=int, default=10, help="Padding size")
    args = vars(ap.parse_args())

    # Carga la imagen en escala de grises
    image = cv2.imread(args["image"], cv2.IMREAD_COLOR_RGB)
    image_gray = cv2.imread(args["image"], cv2.IMREAD_GRAYSCALE)
    
    # Aplica el Filtro padding 
    filtered_image = apply_custom_filter(image_gray, filter, padding=args["padding"])
    
    # Mostrar ambas imágenes en la misma ventana
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Imagen Original")
    axes[0].axis("off")

    axes[1].imshow(image_gray, cmap='gray')
    axes[1].set_title("Imagen en Escala de grises")
    axes[1].axis("off")
    
    axes[2].imshow(filtered_image, cmap='gray')
    axes[2].set_title("Imagen Filtrada")
    axes[2].axis("off")
    
    plt.show()