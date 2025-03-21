import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt




def conv_helper(fragment, kernel):
    """ multiplica 2 matices y devuelve su suma"""
    
    f_row, f_col = fragment.shape
    k_row, k_col = kernel.shape 
    result = 0.0
    for row in range(f_row):
        for col in range(f_col):
            result += fragment[row,col] *  kernel[row,col]
    return result

def convolution(image, kernel):
    """Aplica una convolucion sin padding (valida) de una dimesion 
    y devuelve la matriz resultante de la operación
    """

    image_row, image_col = image.shape #asigna alto y ancho de la imagen 
    kernel_row, kernel_col = kernel.shape #asigna alto y ancho del filtro
   
    output = np.zeros(image.shape) #matriz donde guardo el resultado
   
    for row in range(image_row):
        for col in range(image_col):
                output[row, col] = conv_helper(
                                    image[row:row + kernel_row, 
                                    col:col + kernel_col],kernel)
             

 
    return output


def apply_custom_filter(image, filter, verbose=False):
    """
    Aplica el filtro personalizado a la imagen mediante convolución.
    """
    filtered_image = convolution(image, filter)

    if verbose:
        plt.imshow(filtered_image, cmap='gray')
        plt.title("Imagen Filtrada (Nitidez/Contraste)")
        plt.show()
    
    return filtered_image


if __name__ == '__main__':
    

     # FILTRO (Nitidez/Contraste)
    filter = np.array([[1,    1,  1],  
                       [0,    5,  0],  
                       [-1,  -1,  -1]])  

    # Obtiene la imagen desde la línea de comandos "python script.py -i img.jpg"
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path of image")
    args = vars(ap.parse_args())

    # Carga la imagen en escala de grises
    image = cv2.imread(args["image"], cv2.IMREAD_GRAYSCALE)
    
    plt.imshow(image, cmap='gray')
    plt.title("Imagen Original (En Escala de Grises)")
    plt.show()
    # Aplica el filtro definido
    apply_custom_filter(image, filter, verbose=True)