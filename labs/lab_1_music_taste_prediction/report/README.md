# Lab 1 slides

`build_slides.js` builds the hand-in deck from the figures in `../figures/` (the feature panels are
cut out into `fig/`) and the numbers in `../modelling_results.csv`.

    npm install pptxgenjs
    node build_slides.js            # lab1-slides.pptx, all 25 slides
    node build_slides.js --short    # lab1-slides-15.pptx, the 15 the lab spec says ("say 5-15 slides")

`apply_theme.js` only writes the fonts and colours into the theme part of the pptx.
Speaker notes on every slide hold what to say at the oral.
