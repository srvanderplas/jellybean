library(rvest)
library(xml2)
library(tidyverse)
library(imager)

jb_pages <- "https://www.jellybelly.com/bulk-a-bunch/c/305?pageSize=500"

imgs <- read_html(jb_pages) %>%
  html_nodes("img")

img_data <- tibble(node = imgs, alt = map_chr(imgs, html_attr, "alt"), src = map_chr(imgs, html_attr, "src")) %>%
  filter(str_detect(alt, "[Bb]ulk") & !str_detect(alt, "Assorted|Mix") & str_detect(alt, "Jelly Beans")) %>%
  mutate(src = str_remove(src, "\\?.*$")) %>%
  mutate(img = map(src, load.image)) %>%
  mutate(flavor = str_remove(alt, " Jelly Beans -.*ulk"))

purrr::walk2(img_data$img, paste0("data/", img_data$flavor, ".png"), save.image)

imgs <- tibble(file = list.files("data/"),
               name = str_remove(file, "\\.png"),
               img = map(file.path("data", file), load.image))

# Create image mask
jb_mask <- load.image("inst/JellyBellyMask.png")[,,,1] %>%
  as.cimg() %>%
  erode_square(size = 40)

# This function averages the color over the chunk and reports the results in HSV colorspace instead of RGB
average_color <- function(im) {
  # Convert to HSV color space
  imhsv <- RGBtoHSV(im)

  # Split along color channels
  x <- imsplit(imhsv, "c")

  # Normalize the hue (imager compatibility)
  x[[1]] <- x[[1]]/360

  # Create tibble w/ image and summary stats
  df <- tibble(h = as.numeric(x[[1]]), s = as.numeric(x[[2]]), v = as.numeric(x[[3]]))
  df_mean <- summarize_all(df, c(mean = mean, med = median, var = var)) %>%
    mutate(im = list(imhsv))
}

# This function splits the image and mask up into square chunks
avg_color_chunks <- function(im, mask = jb_mask, nb = 50) {
  # From the mask, we test each chunk to determine if it is completely
  # made up of white pixels. This is stored as an index (mask_chunks)
  mask_chunks <- imsplit(mask, "x", nb = nb) %>%
    purrr::map(., imsplit, "y", nb = nb) %>%
    purrr::map(., ~purrr::map_dbl(., mean) == 1) %>%
    unlist()

  im_chunks <-  imsplit(im, "x", nb = nb) %>%
    purrr::map(., imsplit, "y", nb = nb) %>%
    unlist(recursive = F)

  # Using the index, we keep only chunks of the image
  # (which is the same size as the mask) that correspond to completely
  # white chunks of the mask.

  # Then, the average color of each chunk is calculated as a summary
  tmp <- purrr::map_df(im_chunks[mask_chunks], average_color)
  # average_color returns a tibble containing both the image chunk and the average color
  tmp
}

imgs2 <- imgs %>%
  mutate(avg_colors = purrr::map(img, avg_color_chunks))

# --------------------------------------------------------------
# This bit just plots all of the chunks from each image
# piled on top of each other
img_colors <- imgs2 %>%
  select(-img) %>%
  mutate(img = map(avg_colors, ~imappend(.$im, "y")))

imappend(img_colors$img, "x") %>% plot()

# --------------------------------------------------------------
imgs3 <- imgs2 %>%
  select(-img) %>%
  unnest(avg_colors)

# Plot means - too close together to reasonably separate w/ current method
imgs3 %>%
  select(-file, -im, -matches("med|var")) %>%
  group_by(name) %>%
  summarize_each(mean) %>%
  ggplot(aes(h_mean, s_mean, alpha = v_mean)) + geom_point()

km_opts <- kmeans(imgs3[,c(3:4,6:7)], centers = 63)

imgs_strip <- imgs

# Function to quantize an image into a set number of colors
im_quantize <- function(im, n=256) {

    imdf <- im %>%
      as.data.frame(wide = 'c')
    im_cluster <- imdf %>%
      select(-x, -y) %>%
      kmeans(n)

    im_centers <- im_cluster$centers %>%
      tbl_df %>%
      mutate(label = as.character(row_number()))

    imdf <- imdf %>%
      mutate(label = as.character(im_cluster$cluster)) %>%
      select(x, y, label) %>%
      left_join(im_centers, by = "label") %>%
      select(-label) %>%
      gather(key = 'cc', value = 'value', starts_with("c.")) %>%
      mutate(cc = str_remove(cc, 'c.') %>% as.integer())

    imdf %>% as.cimg(dim = dim(im)) #%>% plot()
  }

format_color <- function(df, px, n_quantize = NULL) {
  im <- df$img

  if (!is.null(n_quantize)) {
    im <- im_quantize(im, n_quantize)
  }
  px <- as.pixset(px)
  tibble(red = R(im)[px],
         green = G(im)[px],
         blue = B(im)[px]) %>%
    na.omit() %>%
    mutate(hex = purrr::pmap_chr(., rgb)) %>%
    group_by(hex) %>% count() %>% ungroup() %>%
    left_join(select(df, -img))
}


jelly_bean_colors <- img_data %>%
  select(img, flavor, src, alt) %>%
  filter(row_number() < 5) %>%
  pmap_df(., format_color, px = jb_mask)

jelly_belly_color_list <- jelly_bean_colors %>%
  # filter(row_number() < 5) %>%
  select(flavor, color_data) %>%
  unnest(color_data) %>%
  mutate(rgb = map(hex, ~col2rgb(.) %>% matrix(ncol = 3) %>% set_colnames(c("r", "g", "b")) %>% as_tibble())) %>%
  unnest_wider(rgb)
