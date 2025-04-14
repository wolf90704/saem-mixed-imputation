function [AUC, Accuracy, Precision, Sensitivity, Specificity, F1_score, Brier_score, Log_score] = binary_metrics(true_labels, predicted_scores, threshold)
    % This function calculates various performance metrics for binary classification.
    % 
    % Inputs:
    %   true_labels - The actual labels (ground truth) of the observations
    %   predicted_scores - The predicted probabilities or scores
    %   threshold - The threshold value for binarizing predicted scores
    %
    % Outputs:
    %   AUC - Area under the ROC curve
    %   Accuracy - Classification accuracy
    %   Precision - Precision of the classifier
    %   Sensitivity - Sensitivity (Recall) of the classifier
    %   Specificity - Specificity of the classifier
    %   F1_score - F1-score, the harmonic mean of precision and sensitivity
    %   Brier_score - Brier score for the predicted probabilities
    %   Log_score - Logarithmic score (cross-entropy)

    % Binarize the predicted scores based on the threshold
    predicted_labels = predicted_scores >= threshold;

    % Calculate True Positives, True Negatives, False Positives, and False Negatives
    TP = sum(true_labels & predicted_labels);  % True Positives
    TN = sum(~true_labels & ~predicted_labels);  % True Negatives
    FP = sum(~true_labels & predicted_labels);  % False Positives
    FN = sum(true_labels & ~predicted_labels);  % False Negatives
    
    % Calculate various performance metrics
    Accuracy = (TP + TN) / (TP + TN + FP + FN);  % Accuracy
    Precision = TP / (TP + FP);  % Precision
    Sensitivity = TP / (TP + FN);  % Sensitivity (Recall)
    Specificity = TN / (TN + FP);  % Specificity
    
    % Calculate F1-score
    F1_score = 2 * (Precision * Sensitivity) / (Precision + Sensitivity);
    
    % Calculate Brier score (mean squared error of predicted probabilities)
    Brier_score = mean((predicted_scores - true_labels).^2);
    
    % Calculate the Log score (cross-entropy loss)
    Log_score = -mean(log(predicted_scores));
    
    % Use the perfcurve function to calculate the Area Under the ROC Curve (AUC)
    [~, ~, ~, AUC] = perfcurve(true_labels, predicted_scores, 1);
end
