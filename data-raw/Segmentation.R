
library(tidyverse)
# library(imager)
#
# im <- load.image("data/Berry Blue.png")
# mask <- load.image("inst/JellyBellyMask.png") %>% channel(1) %>% erode_square(5)
# immask <- imsplit(im, "c") %>% map_il(., ~.*mask) %>% imappend("c")
#
# # Get image gradient
# grad <- imgradient(immask, "xy")
# plot(grad)
#
# # Square the gradient
# gradsq <- grad %>% map_il(~.^2) %>% add()
# plot(gradsq)
#
# # Add the color channels together
# gradsum <- gradsq %>% imsplit("c") %>% add()
# plot(gradsum)
#
# detect.edges <- function(im, sigma=1) {
#   isoblur(im,sigma) %>% imgradient("xy") %>% enorm %>% imsplit("c") %>% add %>% sqrt
# }
#
# imedges <- detect.edges(immask) %T>% plot()
#
# pmap <- 1/(1+imedges)
# plot(pmap)
#
#
# # Pixset segmentation
#
# too_bright <- grayscale(immask) > .6 & mask > .5 #Select pixels with high luminance
# plot(too_bright)
#
#
# useful <- grayscale(immask) < .5 & grayscale(immask) > .4 & mask > .5 #Select pixels with high luminance
# plot(useful)
#
# # Need to figure out how to determine luminance thresholds reasonably...
# # maybe assume ~3% of the unmasked region of the image is not useful due to shine or logo?
#
# brush <- as.cimg(matrix(c(0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0), nrow = 5, byrow = T))
# brushmat <- imappend(imlist(brush, brush, brush), "c")
# seed_points <- useful %>% erode(brush) %>% dilate(brush) %>% erode_square(10)
# plot(seed_points)
#
# seed_points %>% split_connected()
#
#
# # Simple thresholding
# thresh_im <- immask %>% threshold(adjust = 1.5) %T>% plot()
# thresh_im_levels <- thresh_im %>% imsplit("c")  %>% enorm %>% add() %>% sqrt() %T>% plot()
# thresh_im_levels %>% hist()
#
# # pick 2nd most common level
# level <- table(thresh_im_levels) %>% sort(decreasing = T) %>% names() %>% extract2(2) %>% as.numeric()
# as.pixset(thresh_im_levels >= level ) %>% erode_square(80) %T>% plot()
#
#
# ## Trying against all images
#
# mask <- load.image("inst/JellyBellyMask.png") %>% channel(1) %>% erode_square(5)
# imlst <- map_il(list.files("data", "*.png", full.names = T), load.image)
# immask <- map_il(imlst, ~imsplit(., "c") %>% map_il(., ~.*mask) %>% imappend("c"))
#
# thresh_im <- map_il(immask, ~threshold(., adjust = 1.75))
# thresh_im_levels <- map_il(thresh_im, ~imsplit(., "c")  %>% enorm %>% add() %>% sqrt())
#
# # Maybe count separate objects and try several thresholds? Or median object size?



library(EBImage)
mask <- readImage("inst/JellyBellyMask.png")[,,1] %>%
  erode(., kern = makeBrush(35, "disc"))
mask3 <- abind::abind(mask, along = 3) %>%
  abind::abind(., ., ., along = 3) %>%
  Image(colormode = "Color")

imlst <- map(list.files("data", "*.png", full.names = T), readImage)

mask_img <- function(im, mask, value = max(im)) {
  im2 <- im
  im2[!mask] <- value
  im2
}

imlst_mask <- map(imlst, partial(mask_img, mask = mask3))
combine(imlst_mask) %>% plot(all = T)

imlst_gray <- map(imlst_mask, ~channel(., mode = "luminance"))

imlst_thresh <- map(imlst_gray, ~thresh(., w = 30, h = 30, offset = 0.00001))
combine(imlst_thresh) %>% plot(all = T)

imlst_remask <- map(imlst_thresh, ~mask_img(., mask, value = 0))
combine(imlst_remask) %>% plot(all = T)

imlst_label <- map(imlst_remask, bwlabel) %>% map(., colorLabels)
combine(imlst_label[1:9]) %>% plot(all = T)

### Watershed

# Jellybeans are different sizes (picture taken at different distances) in
# different flavors...
# Large ones are ~250 x 180, medium are 200 x 120, small are 150 x 120 --
# but mostly a continuous range between the sizes

# First step: get "background" for adaptive thresholding
# Use a brush that is bigger than the object to detect
# Since our objects vary in size... 250 is probably close enough?

disc <- makeBrush(251, "disc") %>% (function(.) ./sum(.))
imlst_bg <- map(imlst_mask, filter2, filter = disc) %>% map(., partial(mask_img, mask = mask3))
combine(imlst_bg[1:9]) %>% plot(all = T)

# Find all points where the image is more intense than the background plus a fudge factor
imlst_adapt <- map2(imlst_mask, imlst_bg, ~.x > .y + 0.075) # Could make this adaptive based on variation within the image... that would probably make it overall better...
combine(imlst_adapt) %>% plot(all = T)
# At the moment, for some flavors, this gets only the jellybelly label. But that will still work.


# This function should combine the results of each color channel to get a single
# region of intense color
collapse_colors <- function(im) {
  tmp <- getFrames(im) # split apart
  combined_im <- (tmp[[1]] + tmp[[2]] + tmp[[3]]) >= 1 # keep all pixels that are
                                                      # in the highlighted set in at
                                                      # least one color channel
  combined_im
}

imlst_adapt_mask <- map(imlst_adapt, collapse_colors)

clean_pixel_region <- function(im) {
  im %>%
    fillHull() %>%
    erode(makeBrush(5, "disc")) %>%
    dilate(makeBrush(35, "disc")) %>%
    fillHull() %>%
    erode(makeBrush(25, "disc"))
}

# Clean up the pixel sets
highlighted <- map(imlst_adapt_mask, clean_pixel_region)

paint_orig <- map2(highlighted, imlst, ~paintObjects(.x, .y, col = c("black", "white"), opac = c(1, .4)))
combine(paint_orig) %>% plot(all = T)
# Some groups merged - mostly light colors or beans with a lot of shine.
# Need to fix the tabasco image
# Improvements -
# 1. Make the adaptive threshold based on the variability in the image instead of a flat 0.075
# 2. flood fill based on the average of the colors in the image - start at a point closest to the average for flood -filling (that will prevent the label from being an issue)

