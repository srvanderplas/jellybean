library(rvest)
library(xml2)
library(tidyverse)
library(imager)

jb_pages <- "https://www.jellybelly.com/bulk-a-bunch/c/305?pageSize=500"

imgs <- read_html(jb_pages) %>%
  html_nodes("img")

img_data <- tibble(node = imgs, alt = map_chr(imgs, html_attr, "alt"), src = map_chr(imgs, html_attr, "src")) %>%
  filter(str_detect(alt, "10 lbs bulk") & !str_detect(alt, "Assorted|Mix") & str_detect(alt, "Jelly Beans")) %>%
  mutate(src = str_remove(src, "\\?.*$")) %>%
  mutate(img = map(src, load.image)) %>%
  mutate(flavor = str_remove(alt, " Jelly Beans - 10 lbs bulk"))

jb_mask <- load.image("inst/JellyBellyMask.png")[,,,1] %>%
  as.cimg() %>%
  erode_square(size = 40) #%>%
 #add.color()

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
