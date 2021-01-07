import sklearn
from joblib import load
import matplotlib.pyplot as plt
import numpy as np
import pygame

pygame.init()

BLACK = (0, 0, 0)
block_size = 15
WIDTH = 420
HEIGHT = 420
WHITE = (255, 255, 255)
pixels = []

screen = pygame.display.set_mode((WIDTH, HEIGHT))

def drawRect(window, x, y ):
  pygame.draw.rect(window, WHITE, ( x, y, 15, 15))

def drawCircle(window, x, y):
	pygame.draw.circle(window, WHITE, (x, y), 15)

def getPixels():
	for i in range(0, WIDTH, block_size):
		for j in range(0, HEIGHT, block_size):
			pixel = screen.get_at((j, i))[:1]
			pixels.append(pixel)
isPressed = False
run = True


while run:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			run = False
	if event.type == pygame.MOUSEBUTTONDOWN:
	  isPressed = True
	elif event.type == pygame.MOUSEBUTTONUP:
	  isPressed = False
	 
	elif event.type == pygame.MOUSEMOTION and isPressed == True:
		( x, y ) = pygame.mouse.get_pos()
		drawCircle(screen, x, y )
		
	pygame.display.update()

getPixels()
pixels = np.array(pixels)
pixels = pixels.reshape(1,-1)
pygame.quit()


clf = load('digitRecModel.joblib')
predicted = clf.predict(pixels)
print(f"Your number was a {predicted}")



#To plot img

img = pixels.reshape((28, 28))
plt.imshow(img, cmap='gray')
plt.show()
