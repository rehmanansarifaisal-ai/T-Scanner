# Reference MRI library

Place de-identified, permissioned reference images in the corresponding class directory (`glioma`, `meningioma`, `pituitary`, `notumor`). Do not place generated or fabricated patient images here. Optional same-stem `.json` files may include `case_id`, `sequence` and `description` metadata. The app calculates visual HOG-descriptor cosine similarity across the files present.
