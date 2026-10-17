// Lab 1 hand-in deck, built from the figures in labs/lab_1_music_taste_prediction/figures
// and the numbers in modelling_results.csv (5 repeats unless a slide says screen).
//
//   npm install pptxgenjs   (once)
//   node build_slides.js            -> lab1-slides.pptx (all slides)
//   node build_slides.js --short    -> lab1-slides-15.pptx (the 15 the spec says, "say 5-15 slides")
//
// Look: white, Arial, dark text, one grey for captions, one blue for the row that was handed in.
// Panels cut out of feature_overviews.png and decision_tree_roc_auc_best_gain_pairs.png live in fig/.

const pptxgen = require("pptxgenjs");
const { applyTheme } = require("./apply_theme.js"); // fonts and colours into the theme part, same as the deck skeleton

const SHORT = process.argv.includes("--short");
const OUT = SHORT ? "lab1-slides-15.pptx" : "lab1-slides.pptx";

const THEME = {
  name: "Lab1Plain",
  headFontFace: "Arial",
  bodyFontFace: "Arial",
  colors: {
    dk1: "1A1A1A", lt1: "FFFFFF", dk2: "5A5A5A", lt2: "F2F2F2",
    accent1: "4C72B0", accent2: "DD8452", accent3: "8C8C8C", accent4: "C44E52", accent5: "55A868", accent6: "4C72B0",
    hlink: "4C72B0", folHlink: "5A5A5A",
  },
};
const INK = "1A1A1A", GREY = "5A5A5A", BLUE = "4C72B0", RULE = "D9D9D9";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.theme = { headFontFace: "Arial", bodyFontFace: "Arial" };
pres.author = "Andreas Olsson";
pres.title = "Predicting which songs Andreas likes";

const M = 0.5, W = 9, TOP = 1.05, BOT = 4.95;
pres.defineSlideMaster({
  title: "Plain",
  background: { color: "FFFFFF" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.3, w: W, h: 0.6, fontSize: 24, color: INK, align: "left", valign: "top", margin: 0 }, text: "" } },
    { slideNumber: { x: 9.0, y: 5.25, w: 0.5, h: 0.25, fontSize: 9, color: GREY, align: "right", margin: 0 } },
  ],
});
pres.defineSlideMaster({
  title: "Front",
  background: { color: "FFFFFF" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.7, w: 8.4, h: 1.2, fontSize: 32, color: INK, align: "left", valign: "bottom", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.05, w: 8.4, h: 0.6, fontSize: 16, color: GREY, align: "left", valign: "top", margin: 0 }, text: "" } },
  ],
});

// pixel sizes of the figures, so they keep their ratio
const PX = {
  "class_balance.png": [581, 440], "feature_correlations.png": [884, 889],
  "decision_tree_roc_auc_single.png": [696, 461], "decision_tree_roc_auc_pair_gain.png": [767, 461],
  "effect_of_sweeping_different_hyperparams_for_different_methods.png": [1611, 811],
  "learning_curves_for_different_methods.png": [1611, 811],
  "oof_roc_and_confusion_logreg_vs_best.png": [1466, 461],
  "outer_validation_fold_mean_accuracy_all_variants.png": [756, 552],
  "permutation_importances_across_different_methods.png": [885, 554],
  "pair_energy_speechiness.png": [495, 400], "pair_acousticness_speechiness.png": [516, 400],
};
for (const left of ["acousticness", "duration", "instrumentalness", "liveness", "mode", "tempo", "valence"]) PX[`ov_${left}.png`] = [416, 296];
for (const right of ["danceability", "energy", "key", "loudness", "speechiness", "time_signature"]) PX[`ov_${right}.png`] = [395, 296];

// ---- helpers --------------------------------------------------------------------------------
const fit = (name, x, y, w, h, align = "left") => {
  const [iw, ih] = PX[name];
  const s = Math.min(w / iw, h / ih);
  const fw = iw * s, fh = ih * s;
  const fx = align === "center" ? x + (w - fw) / 2 : x;
  return { path: `fig/${name}`, x: fx, y, w: fw, h: fh };
};
const fs = require("fs");
const image = (s, name, x, y, w, h, align) => { const o = fit(name, x, y, w, h, align); if (!fs.existsSync(o.path)) o.path = `../figures/${name}`; s.addImage(o); };
const grid = (s, names, cols, x, y, w, h, gap = 0.1) => {
  const rows = Math.ceil(names.length / cols);
  const cw = (w - gap * (cols - 1)) / cols, ch = (h - gap * (rows - 1)) / rows;
  names.forEach((n, i) => {
    const c = i % cols, r = Math.floor(i / cols);
    image(s, n, x + c * (cw + gap), y + r * (ch + gap), cw, ch, "center");
  });
};
const bullets = (s, items, o = {}) => {
  const runs = items.map((t, i) => ({ text: t, options: { bullet: { indent: 14 }, breakLine: i < items.length - 1 } }));
  s.addText(runs, { x: o.x ?? M, y: o.y ?? TOP, w: o.w ?? W, h: o.h ?? BOT - TOP, fontSize: o.fontSize ?? 14, color: INK,
    align: "left", valign: "top", margin: 0, paraSpaceAfter: o.gap ?? 8, fontFace: "Arial" });
};
const caption = (s, text, o = {}) => s.addText(text, { x: o.x ?? M, y: o.y ?? 4.45, w: o.w ?? W, h: o.h ?? 0.75, fontSize: o.fontSize ?? 12,
  color: GREY, align: "left", valign: "top", margin: 0, fontFace: "Arial" });
const table = (s, header, rows, o = {}) => {
  const cell = (t, extra = {}) => ({ text: String(t), options: { fontSize: o.fontSize ?? 11, color: INK, fontFace: "Arial", margin: o.margin ?? [2, 5, 2, 5],
    border: [{ type: "solid", pt: 0.5, color: RULE }, { type: "none" }, { type: "solid", pt: 0.5, color: RULE }, { type: "none" }], valign: "middle", ...extra } });
  const aligns = o.aligns || header.map(() => "left");
  const data = [header.map((h, j) => cell(h, { bold: true, align: aligns[j], border: [{ type: "none" }, { type: "none" }, { type: "solid", pt: 1, color: INK }, { type: "none" }] }))];
  rows.forEach((r, i) => data.push(r.map((v, j) => cell(v, { align: aligns[j], ...(o.mark && o.mark(r, i) || {}) }))));
  s.addTable(data, { x: o.x ?? M, y: o.y ?? TOP, w: o.w ?? W, colW: o.colW, rowH: o.rowH ?? 0.26, autoPage: false });
};
const slide = (title, notes) => {
  const s = pres.addSlide({ masterName: "Plain" });
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
};

// ---- the slides. keep15 marks the ones that survive --short -----------------------------------
const SLIDES = [];
const add = (id, keep15, build) => SLIDES.push({ id, keep15, build });
let chosen = [];
const ref = (id) => { const i = chosen.findIndex((s) => s.id === id); if (i < 0) throw new Error(`no slide ${id} in this build`); return `slide ${i + 1}`; };

// 1 title
add("title", true, () => {
  const s = pres.addSlide({ masterName: "Front" });
  s.addText("Predicting which songs Andreas likes", { placeholder: "title" });
  s.addText("FRTN65 Lab 1, Andreas Olsson, October 2026", { placeholder: "body" });
  s.addNotes("One sentence version: four tree ensembles tie at 0.83, the rule I wrote first picks bagging as the simplest of them, I handed in the forest on the numerical features for the competition.");
});

// 2 in short
add("short", true, () => {
  const s = slide("In short", `The whole lab in four lines. The override (rf_numeric instead of bagging) is said as an override, ${ref('override')} has the why.`);
  bullets(s, [
    "Data: 736 songs after dropping 14 duplicates, 13 features (10 numerical, 3 categorical), 60% liked",
    "Modelling: 12 methods tried in 51 variants, all through one evaluation protocol",
    "Result: four tree ensembles (xgboost, random forest, bagging, boosting) are indistinguishable on this data, around 0.83 accuracy",
    "The rule I set up before seeing any numbers picks bagging, the simplest of the equally good ones",
    `Handed in: random forest on the 10 numerical features, 0.833 ± 0.018 cross-validated accuracy. Why not bagging is on ${ref('override')}`,
  ], { gap: 10 });
});

// 3 in practice
add("practice", false, () => {
  const s = slide("In practice", "The point of the split: the exploration is allowed to be messy, the comparison is not. The structure is what made 51 variants affordable.");
  bullets(s, [
    "I did the lab in two notebooks instead of a python project: initial_exploration.ipynb (unstructured on purpose) and modelling.ipynb (structured on purpose)",
    "The exploration decides what to try: distributions, correlations, and a cross-validated decision tree screen of which features and feature pairs seem to matter",
    "The modelling decides how to compare, and never changes that mid way. Preprocessing and classifier variants are defined as dicts, and one evaluator runs the same grid search and cross validation on all of them",
    "This made 51 variants super cheap: adding one is one line, and it gets the same folds, tables and plots as everything else",
    "Everything that varies between methods is a dict entry, everything that must not vary (folds, seed, metric, tuning inside the folds) is one function",
  ], { gap: 8 });
});

// 4 the data
add("data", true, () => {
  const s = slide("The data", "The duplicates matter: the same song in both the training and the validation part of a split makes the score look better than it is. The 0.60 floor is what every number is measured from.");
  image(s, "class_balance.png", M, TOP, 4.4, 3.4);
  caption(s, "Class balance in the raw file, 0 = dislike, 1 = like.", { y: 4.5, w: 4.4, h: 0.4 });
  bullets(s, [
    "750 rows, 14 exact duplicates dropped, 736 songs left. No missing values",
    "13 Spotify audio features per song",
    "10 numerical: acousticness, danceability, duration, energy, instrumentalness, liveness, loudness, speechiness, tempo, valence",
    "3 categorical: key, mode, time_signature",
    "443 liked, 293 disliked. Always guessing \"like\" gives 0.60 accuracy, so that is the floor every number is measured from",
  ], { x: 5.3, w: 4.2, gap: 8 });
});

// 5 the four features that matter
add("four", true, () => {
  const s = slide("The data, the four features that matter", "Densities are normalised per class, so the 60/40 does not read as separation. The spike plus tail shape is why robust, quantile and log scaling became variants later.");
  grid(s, ["ov_speechiness.png", "ov_loudness.png", "ov_acousticness.png", "ov_energy.png"], 2, M, TOP, 5.5, 3.9);
  bullets(s, [
    "Liked songs are quiet, acoustic and have almost no speech. Disliked songs are loud, energetic and speechy",
    "The densities are per class, so the 60/40 does not show up as separation here",
    "speechiness: the liked songs are one spike near zero, the disliked ones spread out into a tail. duration, instrumentalness and liveness have the same spike plus tail shape, which is why scaling becomes a question later",
  ], { x: 6.3, w: 3.2, fontSize: 13, gap: 8 });
});

// 6 the categorical ones
add("categorical", false, () => {
  const s = slide("The data, the categorical ones", "key 11 is not more than key 0, hence one-hot. The remaining six numerical features are in the exploration notebook; danceability separates a bit, the rest barely.");
  grid(s, ["ov_key.png", "ov_mode.png", "ov_time_signature.png"], 3, M, TOP, W, 2.1, 0.3);
  bullets(s, [
    "key 11 is not \"more\" than key 0, so these three are treated as qualitative and one-hot encoded (18 columns)",
    `They barely separate the classes here, and it turns out later that no method uses them (${ref('perm')})`,
    "The other six numerical features (danceability, duration, instrumentalness, liveness, tempo, valence) separate a little or not at all, they are in initial_exploration.ipynb",
  ], { y: 3.3, h: 1.65, fontSize: 13, gap: 6 });
});

// 7 correlations
add("corr", true, () => {
  const s = slide("Feature correlations", "Spearman because of the tails: ranks instead of values. Correlation is pairwise and monotone, it cannot see interactions, which is why the tree screen on the next slide exists. Also why a heavily regularised line does well: the main axis is linear and L2 shares the weight over the three correlated columns.");
  image(s, "feature_correlations.png", M, TOP, 3.9, 3.9);
  bullets(s, [
    "Spearman (rank) correlation, bc of the tails. Clustered so the blocks show",
    "energy, loudness and acousticness are one block (energy to loudness 0.85, energy to acousticness −0.74): three columns, one direction",
    "speechiness has the strongest single relation to the label (−0.52) and sits outside that block",
    "tempo, liveness, duration, instrumentalness: near zero against everything",
    "Correlation only sees pairwise monotone relations, not interactions, so the next slide asks a decision tree instead",
  ], { x: 4.8, w: 4.7, fontSize: 13, gap: 8 });
});

// 8 feature screen
add("screen", true, () => {
  const s = slide("Feature screen with decision trees", "Same depth for singles and pairs. I also ran it with repeated folds and a shuffled copy of each feature as a null: the top gains survive, the null never gets above about +0.02, so the small gains at the bottom of the bar chart are noise.");
  image(s, "decision_tree_roc_auc_single.png", M, TOP, 4.4, 3.2);
  image(s, "decision_tree_roc_auc_pair_gain.png", 5.1, TOP, 4.4, 3.2);
  caption(s, "A depth-limited decision tree, 5-fold cross validation, ROC AUC, on each feature alone (left) and on each pair (right). Gain = pair minus the better of its two singles. Every single feature is above 0.5, and every big pair gain has speechiness in it.", { y: 4.1, h: 0.85 });
});

// 9 the two best pairs
add("pairs", false, () => {
  const s = slide("The two best pairs", "This is where the top4 and interactions variants came from. Be ready for: is the gain just extra depth? No, singles and pairs ran at the same depth.");
  image(s, "pair_energy_speechiness.png", M, TOP, 4.3, 3.3);
  image(s, "pair_acousticness_speechiness.png", 5.1, TOP, 4.4, 3.3);
  caption(s, `speechiness splits the songs that the loud/acoustic axis cannot, and the other way round. One model at one depth, so I read it as a hint and not a fact; in the modelling notebook the same question is asked of every method instead (permutation importance, ${ref('perm')}).`, { y: 4.45, h: 0.6 });
});

// 10 preprocessing
add("prep", true, () => {
  const s = slide("Preprocessing", "Rubric point (2): which inputs, qualitative or quantitative, what preprocessing. The pipeline is the whole point: nothing is fit on a validation fold, ever.");
  bullets(s, [
    "All 13 features go in by default: the 10 numerical ones standardised (z-score), the 3 categorical ones one-hot. 28 columns reach the model",
    "Scaling only matters for the methods that measure distances or penalise coefficients (knn, svm, logreg, lda, qda). Trees do not care, so they get the raw columns",
    "Variants on top of that: robust and quantile scaling, log1p on the spiky four, rare levels grouped, categoricals dropped, only the four strong features, pairwise products of the numerical ones",
    "All of it lives inside the sklearn Pipeline, so scaler and encoder are fit on the training part of each fold only, never on the validation part (important)",
    "The test file is read once, at the very end, by the model that was handed in",
  ], { gap: 10 });
});

// 11 the methods
add("methods", true, () => {
  const s = slide("The methods", "Rubric point (1). xgboost is outside the course list, kept as a check on the boosting implementation. Tuned = what the inner grid search picks per split; the count is also what simplicity means later.");
  table(s, ["method", "what it is", "tuned by the grid search"], [
    ["dummy", "always answers \"like\", the floor", "nothing"],
    ["logreg", "one linear boundary, L2 penalty on the coefficients", "C"],
    ["lda", "linear too, gaussian classes with one shared covariance", "nothing"],
    ["qda", "a covariance per class, so a quadratic boundary", "reg_param"],
    ["knn", "vote among the k closest songs in the scaled space", "k, weights"],
    ["tree", "one tree of threshold splits, readable, high variance", "max_depth, min leaf"],
    ["svm_linear", "like logreg but hinge loss, maximises the margin", "C"],
    ["svm_rbf", "kernel trick, a nonlinear boundary", "C, gamma"],
    ["bagging", "300 trees on bootstrap samples, averaged", "min leaf"],
    ["rf", "bagging plus a random feature subset at every split", "max_features, min leaf"],
    ["boosting", "shallow trees in sequence, each fixing the last ones' errors", "learning rate, depth, iterations"],
    ["adaboost", "boosting with stumps and exponential loss", "learning rate, n trees"],
    ["xgboost", "gradient boosting again, outside the course list, a check", "learning rate, depth, min child weight"],
  ], { colW: [1.3, 4.9, 2.8], rowH: 0.26, fontSize: 11 });
});

// 12 the protocol
add("protocol", true, () => {
  const s = slide("The evaluation protocol", "Rubric (2) and (3), the slide the examiner will ask about. Nested loop: for each of 5 outer folds, grid search with 5 inner folds on the outer training part, refit the best, score once on the outer validation fold. Repeated 5 times.");
  bullets(s, [
    "Metric: accuracy, bc that is what the leaderboard measures. Balanced accuracy and ROC AUC are reported beside it but never decided on",
    "Splits: 5-fold stratified cross validation, repeated 5 times with one seed, so 25 outer splits. Every method sees exactly the same 25, so every comparison is paired",
    "Tuning: grid search with 5 inner folds inside the outer training part only. The validation fold never sees the search (nested cross validation)",
    "Uncertainty: the 25 splits share most of their songs, so std / √25 is too small. I use the Nadeau-Bengio corrected standard error, std · √(1/25 + n_val / n_train), on the scores and on the paired differences",
  ], { gap: 12 });
});

// 13 why corrected
add("corrected", false, () => {
  const s = slide("Why the corrected standard error", "Intuition first: the 25 scores are not 25 independent measurements. The 1/4 term is the price of reusing the songs and no amount of repeats pays it off, which is also the answer to why not 10 repeats.");
  bullets(s, [
    "25 scores that come from mostly the same songs are not 25 independent measurements. Treat them as independent and every small difference looks significant",
    "The ordinary variance σ² / n assumes independence. Nadeau and Bengio keep the paired t-test but inflate the variance to (1/n + n_val / n_train) · σ², where σ² is the sample variance of the paired per-split differences",
    "Here n_val / n_train = 1/4, and that term does not shrink with more repeats. 5 repeats give 93% of the precision infinite repeats would (√(1/4) over √(1/25 + 1/4)), so I stopped at 5",
    "51 variants would be over 1000 pairs, so everything is tested against the best one only",
  ], { gap: 12 });
});

// 14 deciding
add("deciding", true, () => {
  const s = slide("Deciding between equally good methods", "Simplicity = fewest tuned hyperparameters, not fewest parameters. Parameter count measures capacity and misleads for ensembles. The order for ties is a stated preference, and here it agrees with the tuned count.");
  bullets(s, [
    "Candidates: every variant within one corrected standard error of the best (the error of the paired gap, not of the means)",
    "Among the candidates: the one with the fewest tuned hyperparameters. Ties by an order fixed in advance: a line < a density model < a distance model < one tree < a kernel < an ensemble of trees",
    "Why fewest tuned and not fewest parameters: among methods the data cannot tell apart, I prefer the one whose score had the fewest chances to look good by search. Parameter count measures capacity, and for ensembles it is a bad guide (more trees do not overfit more)",
    `The band (one error) and the rule were written down before any number came out. ${ref('paired')} shows what two errors, or best mean alone, would have picked instead`,
  ], { gap: 12 });
});

// 15 managing 51 variants
add("managing", false, () => {
  const s = slide("Managing 51 variants", "Screen and promote. The screen shares its first two repeats with the full run, so a promoted variant's full numbers include the screen's splits. Screen numbers never sit next to full numbers in a table.");
  bullets(s, [
    "Running every variant at 5 repeats would take hours, so every variant is first screened at 2 repeats (same seed, so the same first two repeats as the full run)",
    "Promotion, per method: its best variant gets the full 5 repeats, but only if it beats its own base by more than the error of the gap",
    "Screen numbers are never shown next to full numbers. 18 variants ended up at 5 repeats",
    "Every outer split is one row in modelling_results.csv (variant, repeats, fold, scores, the params the inner search picked), so the notebook loads the cache instead of retraining, and every table and plot regenerates from it",
    "Hardware for the record: python 3.12, scikit-learn 1.9, 22 cores with n_jobs = -1",
  ], { gap: 10 });
});

// 16 what the grid search chose
add("gridchose", false, () => {
  const s = slide("What the grid search chose", "Validation curves on the screen folds, one knob at a time with the rest at defaults. Modal picks over the 25 splits: logreg C=0.01 (13/25), svm_linear C=0.01 (19/25), svm_rbf C=1 gamma=0.1 (8/25), knn k=11 distance, tree depth 2, bagging leaf 3 (9/25), rf max_features 0.25 leaf 1 (10/25), xgboost lr 0.03 depth 3.");
  image(s, "effect_of_sweeping_different_hyperparams_for_different_methods.png", M, TOP, W, 3.25, "center");
  caption(s, "One hyperparameter at a time with the rest at defaults, on the screen folds. The linear models want heavy shrinkage (C = 0.01 picked on 13 of 25 splits for logreg and 19 of 25 for the linear svm). knn's validation line is flat from k = 5 to 50 while the training line collapses. The rbf gamma is flat and then falls off a cliff at 1 (every song its own island). Trees want to be shallow alone (depth 2) and deep when averaged (rf: leaf 1, a quarter of the features per split).", { y: 4.35, h: 0.9, fontSize: 11 });
});

// 17 every method on the same splits
add("boxes", false, () => {
  const s = slide("Every method on the same 25 splits", "Read the boxes, not the means: from xgboost down to logreg they overlap. The spread is mostly which songs landed in which fold, which every method shares, and the paired test removes that part.");
  image(s, "outer_validation_fold_mean_accuracy_all_variants.png", M, TOP, 5.4, 3.9);
  bullets(s, [
    "One box per variant: its 25 validation accuracies, best mean on top",
    "Every real method is 18 to 23 points above the dummy",
    "From xgboost down to logreg the boxes overlap. Most of that spread is which songs landed in which fold, and every method shares that",
    "Which is why the comparison has to be paired (next two slides)",
  ], { x: 6.2, w: 3.3, fontSize: 13, gap: 10 });
});

// 18 results table
add("results", true, () => {
  const s = slide("Results, 5 repeats", "Rubric (3). Mean over the 25 outer splits, corrected standard error. Balanced accuracy and ROC AUC are there to read, not to decide on. Dummy: 0.602.");
  const rows = [
    ["xgboost", "0.834 ± 0.016", "0.824", "0.905", "3"],
    ["rf_numeric", "0.833 ± 0.018", "0.825", "0.912", "2"],
    ["rf", "0.833 ± 0.018", "0.825", "0.911", "2"],
    ["bagging", "0.828 ± 0.017", "0.819", "0.905", "1"],
    ["boosting", "0.826 ± 0.015", "0.814", "0.902", "3"],
    ["svm_rbf_top4", "0.824 ± 0.016", "0.808", "0.878", "2"],
    ["knn_top4", "0.824 ± 0.015", "0.805", "0.880", "2"],
    ["svm_rbf_numeric", "0.821 ± 0.016", "0.804", "0.881", "2"],
    ["logreg_elastic", "0.817 ± 0.018", "0.805", "0.888", "1"],
    ["svm_rbf", "0.814 ± 0.016", "0.801", "0.887", "2"],
    ["svm_linear", "0.813 ± 0.017", "0.800", "0.888", "1"],
    ["adaboost", "0.809 ± 0.017", "0.796", "0.890", "2"],
    ["logreg", "0.807 ± 0.019", "0.794", "0.887", "1"],
    ["knn", "0.802 ± 0.012", "0.776", "0.883", "2"],
    ["tree", "0.800 ± 0.017", "0.786", "0.845", "2"],
    ["lda", "0.795 ± 0.016", "0.782", "0.879", "0"],
    ["qda", "0.783 ± 0.017", "0.788", "0.879", "1"],
  ];
  table(s, ["variant", "accuracy ± corrected error", "balanced accuracy", "ROC AUC", "tuned"], rows,
    { colW: [2.3, 2.3, 1.7, 1.4, 1.3], rowH: 0.186, fontSize: 9.5, margin: [1, 5, 1, 5], aligns: ["left", "right", "right", "right", "right"],
      mark: (r) => r[0] === "rf_numeric" ? { color: BLUE, bold: true } : (r[0] === "bagging" ? { bold: true } : {}) });
  caption(s, "Mean over the 25 outer splits ± Nadeau-Bengio corrected standard error. Bold: what the rule picks. Blue: what was handed in. dummy (always \"like\"): 0.602.", { y: 4.72, h: 0.3, fontSize: 11 });
});

// 19 paired against the best
add("paired", true, () => {
  const s = slide("Paired against the best, and the rule applied", "Per split difference to xgboost, corrected error on the differences. Seven candidates inside one error; bagging tunes one hyperparameter, the others two or three. Two errors would let logreg_elastic in, and the rule would take it (one tuned, and a line comes first in the order).");
  table(s, ["variant", "gap to xgboost", "corrected error", "within 1 error", "within 2 errors"], [
    ["rf_numeric", "−0.001", "0.007", "yes", "yes"],
    ["rf", "−0.001", "0.007", "yes", "yes"],
    ["bagging", "−0.006", "0.008", "yes", "yes"],
    ["boosting", "−0.008", "0.009", "yes", "yes"],
    ["svm_rbf_top4", "−0.010", "0.013", "yes", "yes"],
    ["knn_top4", "−0.010", "0.016", "yes", "yes"],
    ["svm_rbf_numeric", "−0.013", "0.013", "no", "yes"],
    ["logreg_elastic", "−0.017", "0.011", "no", "yes"],
    ["svm_rbf", "−0.020", "0.011", "no", "yes"],
    ["svm_linear", "−0.021", "0.010", "no", "no"],
    ["logreg", "−0.027", "0.012", "no", "no"],
    ["lda", "−0.039", "0.011", "no", "no"],
  ], { colW: [2.3, 1.7, 1.7, 1.65, 1.65], rowH: 0.235, fontSize: 11, aligns: ["left", "right", "right", "center", "center"],
      mark: (r) => r[0] === "rf_numeric" ? { color: BLUE, bold: true } : (r[0] === "bagging" ? { bold: true } : {}) });
  caption(s, "The rule (one error, fewest tuned hyperparameters) picks bagging: seven candidates, bagging tunes one hyperparameter. At two errors logreg_elastic joins and the rule would pick it instead. Best mean alone would pick xgboost.", { y: 4.3, h: 0.65, fontSize: 11 });
});

// 20 the override
add("override", true, () => {
  const s = slide("Why I handed in rf_numeric and not bagging", "Say it as an override. Under the rule: bagging. For the competition: the forest without the three columns no model uses. Both strings are in the zip. If asked why not xgboost: not a course method, and not better than the forest by anything measurable.");
  bullets(s, [
    "Under the rule: bagging. That string is submission_best_choice.txt",
    "The leaderboard counts accuracy only, so for the competition I took the highest expected score among the methods the data cannot tell apart, and gambled",
    "rf_numeric is the forest without key, mode and time_signature. Same mean as the full forest (gap +0.000 ± 0.007 on the screen, 0.833 for both at 5 repeats) and the best balanced accuracy and ROC AUC of all the course methods (0.825, 0.912)",
    `Dropping the three categorical columns costs nothing the data can detect, and every diagnostic rates them at zero (${ref('perm')})`,
    "Not xgboost: outside the course list, and not better than the forest by anything measurable",
    "The two strings differ on 10 of the 200 test songs",
  ], { gap: 8 });
});

// 21 learning curves
add("learning", false, () => {
  const s = slide("Learning curves", "The bias-variance motivation for the ensembles in one picture. Training accuracy 1.0 is not a bug: at their defaults tree, rf and boosting grow until the leaves are pure.");
  image(s, "learning_curves_for_different_methods.png", M, TOP, W, 3.25, "center");
  caption(s, "Bias on the left, variance on the right. Train and validation accuracy against training set size, defaults, screen folds. tree, rf and boosting train at 1.0 bc at their defaults they fit until the leaves are pure, so read the validation line. logreg: the two lines have met, bias limited, more data will not help. tree: the biggest gap, variance limited, 0.76. rf: the same memorised training set, validation at 0.84 and still climbing at 590 songs. Averaging bought eight points from the same kind of tree.", { y: 4.35, h: 0.9, fontSize: 11 });
});

// 22 permutation importance
add("perm", true, () => {
  const s = slide("What every method leans on", "Permutation importance on held-out folds: shuffle one column, see how much accuracy drops. The bottom three rows are the evidence for rf_numeric.");
  image(s, "permutation_importances_across_different_methods.png", M, TOP, 5.6, 3.9);
  bullets(s, [
    "Permutation importance: the drop in held-out accuracy when one feature is shuffled, per method",
    "speechiness is the one feature every method leans on (0.09 to 0.11), then acousticness and loudness",
    "key, mode and time_signature are zero for every method. This is the evidence for dropping them in rf_numeric",
    "knn and boosting also use danceability a bit. Nobody uses tempo, valence or liveness",
  ], { x: 6.3, w: 3.2, fontSize: 13, gap: 10 });
});

// 23 what the variants taught
add("variants", false, () => {
  const s = slide("What the variants taught", "Screen numbers, 2 repeats, gap to the method's own base on the same splits. These are the regularisation, scaling and feature engineering motivations, each with a number.");
  bullets(s, [
    "Feature choice: knn on the four strong features +0.022 ± 0.013, the 18 one-hot axes drown the distance. The rbf svm on the four: +0.020, on the numerical ones only: +0.019",
    "Regularisation: L1 and elastic net logreg +0.018 / +0.019 over L2. Pairwise products alone +0.005, with L1 on them +0.017, so the products need the penalty",
    "Scaling: robust scaling costs knn and the rbf svm four points (−0.039, −0.041), the IQR of the spiky columns is tiny so the tails explode. log1p on the spiky four helps logreg +0.015. Trees: tree_noscale equals tree split for split",
    "Ensembles: no variant of rf or xgboost helped (−0.001 to −0.010), they were already fine",
  ], { h: 3.45, gap: 10 });
  caption(s, "Screen numbers (2 repeats), gap to the method's own base on the same splits ± corrected error.", { y: 4.6, h: 0.35, fontSize: 11 });
});

// 24 production model
add("production", false, () => {
  const s = slide("The production model", "Refit once on all 736 songs with the same grid search. The honest range: 0.833 ± 0.018 cross-validated, and 200 test songs is a small set, so 0.80 to 0.86.");
  image(s, "oof_roc_and_confusion_logreg_vs_best.png", M, 1.0, 8.6, 2.7, "center");
  bullets(s, [
    "rf_numeric refit on all 736 songs with the same grid search (300 trees, a quarter of the features per split, min leaf 1). 128 of the 200 test songs predicted as liked",
    "Out-of-fold ROC and confusion, logreg against bagging (rf_numeric is within a hair of bagging): the ensemble's gain over the line is mostly on the disliked songs, 227 right against 208",
    "What to expect on the 200 songs: 0.80 to 0.86",
  ], { y: 3.8, h: 1.2, fontSize: 12, gap: 4 });
});

// 25 conclusions and disclosure
add("conclusions", true, () => {
  const s = slide("Conclusions", "Rubric (4). Then the disclosure, said plainly. Zip: both notebooks executed, modelling_results.csv, figures/, both submission strings, a short README.");
  bullets(s, [
    "Four tree ensembles tie at 0.83 and nothing beats them. By the rule the simplest of them, bagging, wins. For the competition, the forest on the numerical features",
    "The data has one strong axis (loud and energetic against quiet and acoustic) plus speechiness. A heavily regularised line already gets 0.81, pairs and trees buy the last two points",
    "Scaling only matters where distances or penalties are, and the wrong scaler costs four points. The three categorical columns carry nothing",
    "Pairing the splits and correcting the standard error is what made \"equally good\" a statement instead of a feeling",
  ], { h: 3.0, gap: 8 });
  s.addText("AI (Claude) was used to understand how professional ML projects structure the explore, protocol and compare steps, for help with some of the plotting code, and for the paired comparison logic (corrected standard error, decision rule). The exploration, the modelling notebook, the variant choices, the runs and the conclusions are my own.",
    { x: M, y: 4.2, w: W, h: 0.8, fontSize: 11, color: GREY, align: "left", valign: "top", margin: 0, fontFace: "Arial" });
});

// ---- build ------------------------------------------------------------------------------------
chosen = SLIDES.filter((s) => !SHORT || s.keep15);
chosen.forEach((s) => s.build());
(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log(`${OUT}: ${chosen.length} slides`);
})();
