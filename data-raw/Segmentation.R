library(imager)
library(tidyverse)

im <- load.image("data/Berry Blue.png")
mask <- load.image("inst/JellyBellyMask.png") %>% channel(1) %>% erode_square(5)
immask <- imsplit(im, "c") %>% map_il(., ~.*mask) %>% imappend("c")

# Get image gradient
grad <- imgradient(immask, "xy")
plot(grad)

# Square the gradient
gradsq <- grad %>% map_il(~.^2) %>% add()
plot(gradsq)

# Add the color channels together
gradsum <- gradsq %>% imsplit("c") %>% add()
plot(gradsum)

detect.edges <- function(im, sigma=1) {
  isoblur(im,sigma) %>% imgradient("xy") %>% enorm %>% imsplit("c") %>% add %>% sqrt
}

imedges <- detect.edges(immask) %T>% plot()

pmap <- 1/(1+imedges)
plot(pmap)


# Pixset segmentation

too_bright <- grayscale(immask) > .6 & mask > .5 #Select pixels with high luminance
plot(too_bright)


useful <- grayscale(immask) < .5 & grayscale(immask) > .4 & mask > .5 #Select pixels with high luminance
plot(useful)

# Need to figure out how to determine luminance thresholds reasonably...
# maybe assume ~3% of the unmasked region of the image is not useful due to shine or logo?

brush <- as.cimg(matrix(c(0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0), nrow = 5, byrow = T))
brushmat <- imappend(imlist(brush, brush, brush), "c")
seed_points <- useful %>% erode(brush) %>% dilate(brush) %>% erode_square(10)
plot(seed_points)

seed_points %>% split_connected()


# Simple thresholding
thresh_im <- immask %>% threshold(adjust = 1.5) %T>% plot()
thresh_im_levels <- thresh_im %>% imsplit("c")  %>% enorm %>% add() %>% sqrt() %T>% plot()
thresh_im_levels %>% hist()

# pick 2nd most common level
level <- table(thresh_im_levels) %>% sort(decreasing = T) %>% names() %>% extract2(2) %>% as.numeric()
as.pixset(thresh_im_levels >= level ) %>% erode_square(80) %T>% plot()


## Trying against all images

mask <- load.image("inst/JellyBellyMask.png") %>% channel(1) %>% erode_square(5)
imlst <- map_il(list.files("data", "*.png", full.names = T), load.image)
immask <- map_il(imlst, ~imsplit(., "c") %>% map_il(., ~.*mask) %>% imappend("c"))

thresh_im <- map_il(immask, ~threshold(., adjust = 1.75))
thresh_im_levels <- map_il(thresh_im, ~imsplit(., "c")  %>% enorm %>% add() %>% sqrt())

# Maybe count separate objects and try several thresholds? Or median object size?



library(EBImage)
mask <- readImage("inst/JellyBellyMask.png")[,,1] %>% erode(., kern = makeBrush(35, "disc"))
mask3 <- abind::abind(mask, along = 3) %>% abind::abind(., ., ., along = 3) %>% Image(colormode = "Color")

imlst <- map(list.files("data", "*.png", full.names = T), readImage)

mask_img <- function(im, mask, value = max(im)) {
  im2 <- im
  im2[!mask] <- value
  im2
}

imlst_mask <- map(imlst, partial(mask_img, mask = mask3))

combine(imlst_mask) %>% plot(all = T)

