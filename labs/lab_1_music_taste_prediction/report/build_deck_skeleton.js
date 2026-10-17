const pptxgen = require("pptxgenjs");
const { applyTheme } = require("./apply_theme.js")  // from the pptx skill, copied beside this script;

// the four decisions: one layout, one face in three sizes, one accent taken from the figures, exactness
const THEME = {
  name: "Lab1Plain",
  headFontFace: "Arial",
  bodyFontFace: "Arial",
  colors: {
    dk1: "1A1A1A", lt1: "FFFFFF", dk2: "5A5A5A", lt2: "F2F2F2",
    accent1: "4C72B0",            // seaborn deep blue, the train line in every figure
    accent2: "DD8452", accent3: "8C8C8C", accent4: "C44E52", accent5: "55A868", accent6: "4C72B0",
    hlink: "4C72B0", folHlink: "5A5A5A",
  },
};
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";           // 10 x 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "Andreas Olsson"; pres.title = "FRTN65 Lab 1, music taste prediction";
const C = pres.SchemeColor;
const M = 0.5, W = 10 - 2 * M;         // margin and content width
const TITLE = { name: "title", type: "title", x: M, y: 0.35, w: W, h: 0.6, fontSize: 24, bold: true, color: C.text1, align: "left", margin: 0, valign: "top" };
const CAPTION = { name: "caption", type: "body", x: M, y: 4.85, w: W, h: 0.45, fontSize: 12, color: C.text2, align: "left", margin: 0, valign: "top" };
const FOOT = { text: "FRTN65 Lab 1  ·  Andreas Olsson", options: { x: M, y: 5.32, w: 5, h: 0.22, fontSize: 9, color: C.text2, margin: 0, isTextBox: true } };

// layouts
pres.defineSlideMaster({ title: "Title", background: { color: THEME.colors.lt1 }, objects: [
  { placeholder: { options: { name: "title", type: "title", x: M, y: 1.7, w: W, h: 1.0, fontSize: 32, bold: true, color: C.text1, align: "left", margin: 0 }, text: "Deck title" } },
  { placeholder: { options: { name: "body", type: "body", x: M, y: 2.8, w: W, h: 1.4, fontSize: 16, color: C.text2, align: "left", margin: 0, valign: "top" }, text: "One line result, then course, name, date" } },
]});
pres.defineSlideMaster({ title: "Figure", background: { color: THEME.colors.lt1 }, objects: [
  { placeholder: { options: TITLE, text: "One claim, as a sentence" } },
  { placeholder: { options: { name: "figure", type: "pic", x: M, y: 1.05, w: W, h: 3.7, color: C.text1 }, text: "Drop the figure here" } },
  { placeholder: { options: CAPTION, text: "One sentence saying what to see in it" } },
  { text: FOOT.text, options: { ...FOOT.options } },
  { slideNumber: { x: 9.0, y: 5.32, w: 0.5, h: 0.22, fontSize: 9, color: C.text2, align: "right", margin: 0 } },
]});
pres.defineSlideMaster({ title: "Figure and text", background: { color: THEME.colors.lt1 }, objects: [
  { placeholder: { options: TITLE, text: "One claim, as a sentence" } },
  { placeholder: { options: { name: "figure", type: "pic", x: M, y: 1.05, w: 5.6, h: 3.7, color: C.text1 }, text: "Drop the figure here" } },
  { placeholder: { options: { name: "body", type: "body", x: 6.4, y: 1.05, w: 3.1, h: 3.7, fontSize: 14, color: C.text1, align: "left", margin: 0, valign: "top", paraSpaceAfter: 8 }, text: "Three short lines at most" } },
  { placeholder: { options: CAPTION, text: "One sentence saying what to see in it" } },
  { text: FOOT.text, options: { ...FOOT.options } },
  { slideNumber: { x: 9.0, y: 5.32, w: 0.5, h: 0.22, fontSize: 9, color: C.text2, align: "right", margin: 0 } },
]});
pres.defineSlideMaster({ title: "Table", background: { color: THEME.colors.lt1 }, objects: [
  { placeholder: { options: TITLE, text: "One claim, as a sentence" } },
  { placeholder: { options: { name: "table", type: "tbl", x: M, y: 1.05, w: W, h: 3.7, color: C.text1 }, text: "Table: typeset, right-aligned numbers, the chosen row in the accent" } },
  { placeholder: { options: CAPTION, text: "One sentence saying what to see in it" } },
  { text: FOOT.text, options: { ...FOOT.options } },
  { slideNumber: { x: 9.0, y: 5.32, w: 0.5, h: 0.22, fontSize: 9, color: C.text2, align: "right", margin: 0 } },
]});
pres.defineSlideMaster({ title: "Text", background: { color: THEME.colors.lt1 }, objects: [
  { placeholder: { options: TITLE, text: "One claim, as a sentence" } },
  { placeholder: { options: { name: "body", type: "body", x: M, y: 1.05, w: W, h: 3.7, fontSize: 16, color: C.text1, align: "left", margin: 0, valign: "top", paraSpaceAfter: 10 }, text: "Short lines, no sub-bullets" } },
  { placeholder: { options: CAPTION, text: "One sentence saying what to take from it" } },
  { text: FOOT.text, options: { ...FOOT.options } },
  { slideNumber: { x: 9.0, y: 5.32, w: 0.5, h: 0.22, fontSize: 9, color: C.text2, align: "right", margin: 0 } },
]});

const fig = (name, w, h, x, y) => ({ path: name, x, y, w, h });
const fit = (name, boxW, boxH, x0, y0, iw, ih) => { // fit an image into the figure box, keep ratio, left/top aligned
  const s = Math.min(boxW / iw, boxH / ih); return { path: name, x: x0, y: y0, w: iw * s, h: ih * s }; };
const add = (master, title, caption, notes, body) => {
  const s = pres.addSlide({ masterName: master });
  s.addText(title, { placeholder: "title" });
  if (caption) s.addText(caption, { placeholder: "caption" });
  if (body) s.addText(body, { placeholder: "body" });
  if (notes) s.addNotes(notes);
  return s;
};

// 1
let s = pres.addSlide({ masterName: "Title" });
s.addText("Predicting which songs Andreas likes", { placeholder: "title" });
s.addText([{ text: "736 songs, 51 variants of 12 methods through one protocol, chosen method: <name> at <acc> ± <se>", options: { breakLine: true } },
           { text: "FRTN65 Lab 1  ·  Andreas Olsson  ·  October 2026" }], { placeholder: "body" });
s.addNotes("Say the result, then say in words what the rule did: the four ensembles are equivalent, the rule picked the simplest of them.");
// 2
add("Figure", "The data runs along one axis: quiet, acoustic, wordless songs are liked", "Densities normalised within class. Liked songs are quiet, acoustic and nearly free of speech; the discrete columns barely separate.",
  "From initial_exploration: single_feature_plots cropped to speechiness, loudness, acousticness, energy and one discrete column. Mention the 14 duplicate rows dropped before any split.");
// 3
add("Figure and text", "Feature pairs carry more than their parts", "Gain = pair ROC AUC minus the better single, depth-4 trees under cross validation.",
  "From initial_exploration: the pair gain bars and the energy + speechiness plot. The gain survives equal depth, repeated folds and a shuffled-copy null, which never exceeds +0.02.",
  [{ text: "energy + speechiness, acousticness + speechiness: about +0.08", options: { bullet: true, breakLine: true } },
   { text: "small gains sit inside the null", options: { bullet: true, breakLine: true } },
   { text: "so a model that can bend along pairs should gain", options: { bullet: true } }]);
// 4
add("Table", "(1) Twelve methods, what each does and what is tuned", "xgboost is outside the course list, kept as a check on the boosting implementation.",
  "From METHODS and the SIMPLICITY comments: two columns, what it does, what is tuned. Twelve rows including dummy.");
// 5
add("Text", "(2) How the inputs went in", "With the defaults, 28 columns reach the model: 10 scaled numeric, 18 one-hot.",
  "From make_preprocessor and the Data cell. Say why key/mode/time_signature are qualitative, what was tried on the spiky four, and that preprocessing is fit inside each fold.",
  [{ text: "all 13 features; 10 numeric standardised", options: { bullet: true, breakLine: true } },
   { text: "key, mode, time_signature one-hot: key 11 is not more than key 0", options: { bullet: true, breakLine: true } },
   { text: "spiky four (duration, speechiness, instrumentalness, liveness): log1p, robust and quantile scaling tried as variants", options: { bullet: true, breakLine: true } },
   { text: "14 exact duplicates dropped before any split", options: { bullet: true, breakLine: true } },
   { text: "scaling and encoding live inside the pipeline, fit on the training part only", options: { bullet: true } }]);
// 6
add("Text", "(2) The protocol, written before any number", "Every method is scored on the same 25 splits, so every comparison is paired.",
  "Lift the Decisions markdown cell: accuracy primary because the leaderboard scores it, 5x5 stratified outer splits, grid search inside each outer training part, Nadeau-Bengio corrected errors, the decision rule, screen (2 repeats) versus full (5) and the promotion rule. Have the nested-loop pseudocode ready.",
  [{ text: "metric: accuracy (the leaderboard); balanced accuracy and AUC reported beside", options: { bullet: true, breakLine: true } },
   { text: "5-fold stratified, repeated 5 times, one seed: 25 paired outer splits", options: { bullet: true, breakLine: true } },
   { text: "tuning: grid search inside each outer training part (5 inner folds)", options: { bullet: true, breakLine: true } },
   { text: "error bars: std · sqrt(1/k + n_val/n_train), the Nadeau-Bengio correction", options: { bullet: true, breakLine: true } },
   { text: "rule: within one corrected error of the best, take the fewest tuned hyperparameters", options: { bullet: true, breakLine: true } },
   { text: "screen variants at 2 repeats; promote to 5 only if a variant beats its base by more than the error of the gap", options: { bullet: true } }]);
// 7
s = add("Figure", "(2) What the searches chose, and why", "Linear models want heavy shrinkage (C = 0.01); kNN's validation line is flat; the RBF kernel falls off a cliff at gamma = 1.",
  "Table of modal hyperparameters from the params column beside this figure, or on a build. logreg C=0.01 on 13/25, svm_linear C=0.01 on 19/25, tree depth 2, bagging leaf 3, rf max_features 0.25.");
s.addImage(fit("effect_of_sweeping_different_hyperparams_for_different_methods.png", W, 3.7, M, 1.05, 1600, 800));
// 8
s = add("Figure", "(3) Every method on the same 25 splits", "Each box is one method's 25 validation accuracies; from xgboost down to logistic regression the boxes overlap, so the comparison has to be paired.",
  "The five-repeat summarize table goes on a build or the next slide: mean ± corrected error, balanced accuracy, AUC.");
s.addImage(fit("outer_validation_fold_mean_accuracy_all_variants.png", W, 3.7, M, 1.05, 750, 550));
// 9
add("Table", "(3) Paired against the best, and the rule applied", "Inside one corrected error: xgboost, rf, bagging, boosting, knn_top4. Fewest tuned hyperparameters among them: bagging. Best mean alone would pick xgboost.",
  "From paired_vs_best and decide: gap, se, p, best-wins. Say the strictness was fixed in advance and what two errors or best-mean would have chosen. Rerun after the SVM top4 / numeric variants get their full run.");
// 10
add("Text", "(3) What the variants taught", "Screen numbers (2 repeats), named as such.",
  "From the screen table. Each bullet is one motivation the rubric asks for, with a number.",
  [{ text: "kNN on the four strong features gains two points over kNN on all 13: the one-hot axes drown the distance", options: { bullet: true, breakLine: true } },
   { text: "L1 and elastic net gain about two points over L2; L1 keeps the same few features", options: { bullet: true, breakLine: true } },
   { text: "robust scaling costs kNN and the RBF SVM four points: a tiny IQR blows the tails up", options: { bullet: true, breakLine: true } },
   { text: "scaling changes nothing for trees: tree_noscale equals tree split for split", options: { bullet: true } }]);
// 11
s = add("Figure and text", "(3) Why: bias on the left, variance on the right", "Training accuracy of the tree family is 1.0 at the defaults because they fit until pure; read the validation line.",
  "Learning curves left, permutation heatmap in the text column or on a build. Logistic regression bias limited (lines met), the lone tree variance limited, the forest still climbing at 590 songs. Speechiness is the one feature every method leans on; key, mode and time_signature are zero for all.",
  [{ text: "logreg: lines have met, bias limited", options: { bullet: true, breakLine: true } },
   { text: "tree: biggest gap, variance limited", options: { bullet: true, breakLine: true } },
   { text: "rf: still climbing at 590 songs", options: { bullet: true, breakLine: true } },
   { text: "every method leans on speechiness first; key, mode, meter are zero", options: { bullet: true } }]);
s.addImage(fit("learning_curves_for_different_methods.png", 5.6, 3.7, M, 1.05, 1600, 800));
// 12
s = add("Figure", "(4) Conclusion: <chosen method>, refit once, 200 songs predicted", "Out-of-fold confusion: the gain over logistic regression is mostly on disliked songs (227 vs 208 right). Expect 0.80 to 0.86 on the 200 songs.",
  "Refit setting, the submission file, the honest range, and one line on the code: two notebooks, definitions up top, every number regenerable from the seed and the results CSV.");
s.addImage(fit("oof_roc_and_confusion_logreg_vs_best.png", W, 3.7, M, 1.05, 1500, 450));

(async () => {
  await pres.writeFile({ fileName: "lab1-deck-skeleton.pptx" });
  await applyTheme("lab1-deck-skeleton.pptx", THEME);
  console.log("written");
})();
