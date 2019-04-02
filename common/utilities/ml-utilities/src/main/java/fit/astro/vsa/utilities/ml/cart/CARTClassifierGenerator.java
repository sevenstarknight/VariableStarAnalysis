/*
 * Copyright (C) 2016 Kyle Johnston
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package fit.astro.vsa.utilities.ml.cart;

import fit.astro.vsa.common.bindings.ml.ClassificationResult;
import fit.astro.vsa.common.datahandling.LabelHandling;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.math3.linear.RealVector;

/**
 * Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984).
 * Classification and regression trees. CRC press.
 *
 * Loh, W. Y. (2011). Classification and regression trees. Wiley
 * Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 1(1), 14-23.
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class CARTClassifierGenerator {

    // ============== Input
    private final Map<Integer, RealVector> setOfTrainingData;
    private final Map<Integer, String> setOfTrainingClasses;

    // ============== Internal
    private int dimensions;

    private ImpurityType impurityType = ImpurityType.GDI;
    private int minNodeSize = 5;
    private double alpha;

    /**
     * Initialize CART Generator
     * @param setOfTrainingData
     * @param setOfTrainingClasses
     */
    public CARTClassifierGenerator(
            Map<Integer, RealVector> setOfTrainingData,
            Map<Integer, String> setOfTrainingClasses) {
        this.setOfTrainingData = setOfTrainingData;
        this.setOfTrainingClasses = setOfTrainingClasses;
    }

    /**
     * Make the CART (Binary Search Tree)
     * @param alpha
     * @param crossvalPatternMap
     * @param crossvalClassMap
     * @return
     */
    public CART generateCART(double alpha,
            Map<Integer, RealVector> crossvalPatternMap,
            Map<Integer, String> crossvalClassMap) {

        dimensions = setOfTrainingData.values().iterator().next().getDimension();

        // ========================================
        // Set up: Split Input Patterns Into Class Groups
        Map<String, Map<Integer, RealVector>> classMembers
                = LabelHandling.sortIntoMaps(setOfTrainingData, setOfTrainingClasses);

        // Initialize Prior Probabilities
        Map<String, Double> priorProbability = new HashMap<>();
        for (Entry<String, Map<Integer, RealVector>> entry : classMembers.entrySet()) {
            double sizeOfList = (double) entry.getValue().size();
            double priorProb = sizeOfList / (double) setOfTrainingClasses.size();
            priorProbability.put(entry.getKey(), priorProb);
        }

        CARTPruneAndGrow cartpag = new CARTPruneAndGrow(setOfTrainingData,
                setOfTrainingClasses, dimensions, impurityType);

        cartpag.setMinNodeSize(minNodeSize);

        //=========================================================
        // Grow Tree
        CART cart = cartpag.GrowCART(impurityType);

        // ================================================================
        // Prune Tree
        cart = cartpag.PruneCART(alpha, cart, crossvalPatternMap, crossvalClassMap);

        return cart;
    }

    //======================================================================
    // Use CART Object
    /**
     *
     * @param cart
     * @param inputPatternMap
     * <p>
     * @return
     */
    public static ClassificationResult execute(CART cart,
            Map<Integer, RealVector> inputPatternMap) {

        Map<Integer, String> labelEstimate = new HashMap<>();
        Map<Integer, Map<String, Double>> labelPostProb = new HashMap<>();

        // ========================================        
        // Loop Over Cross-Validation Data
        for (Entry<Integer, RealVector> entry : inputPatternMap.entrySet()) {

            RealVector crossvalPattern = entry.getValue();

            String classEst
                    = estimateClassLabel(cart, crossvalPattern);

            Map<String, Double> probEst = estimatePostProb(cart,
                    crossvalPattern);

            labelEstimate.put(entry.getKey(), classEst);
            labelPostProb.put(entry.getKey(), probEst);

        }

        return new ClassificationResult(labelEstimate, labelPostProb,
                cart.getUniqueLabelCount());
    }

    /**
     *
     * @param cart
     * @param inputPatternMap
     * @param inputClassMap
     * <p>
     * @return
     */
    public static double estimateMisclassificationRate(
            CART cart,
            Map<Integer, RealVector> inputPatternMap,
            Map<Integer, String> inputClassMap) {
        double misclassifications = 0;

        for (Entry<Integer, RealVector> entry : inputPatternMap.entrySet()) {
            String labelEst = estimateClassLabel(cart, entry.getValue());

            if (!inputClassMap.get(entry.getKey()).equalsIgnoreCase(labelEst)) {
                // Terminal Node Found 
                misclassifications++;
            }
        }

        return misclassifications / (double) inputPatternMap.size();
    }

    /**
     * To run the classifier, the whole training dataset is not needed, only the
     * CART objected generated by the training operation.
     * <p>
     * @param pattern
     * @param cart
     * <p>
     * @return
     */
    public static String estimateClassLabel(CART cart,
            RealVector pattern) {
        Boolean runningCART = Boolean.TRUE;
        String label = "Missing";
        Integer currentIdx = 1;

        while (runningCART) {

            if (cart.getDecisionNodes().containsKey(currentIdx)) {
                // Is a Decision

                ImmutablePair<Integer, Double> decisionPair
                        = cart.getDecisionNodes().get(currentIdx);

                ImmutablePair<Integer, Integer> lrPair 
                        = cart.getTree().get(currentIdx);

                if (pattern.getEntry(decisionPair.getKey())
                        > decisionPair.getValue()) {
                    // Left
                    currentIdx = lrPair.getKey();
                } else {
                    //Right
                    currentIdx = lrPair.getValue();
                }

            } else {
                label = cart.getTerminalNodes().get(currentIdx);
                // Terminal Node Found 
                break;
            }
        }

        return label;
    }

    /**
     * Estimate the Posterior Probability
     *
     * @param cart
     * @param pattern
     * @return
     */
    public static Map<String, Double> estimatePostProb(
            CART cart,
            RealVector pattern) {
        
        Boolean runningCART = Boolean.TRUE;
        Map<String, Double> postProb = new HashMap<>();
        postProb.put("Missed", 1.0);
        Integer currentIdx = 1;

        while (runningCART) {

            if (cart.getDecisionNodes().containsKey(currentIdx)) {
                // Is a Decision

                ImmutablePair<Integer, Double> decisionPair
                        = cart.getDecisionNodes().get(currentIdx);

                ImmutablePair<Integer, Integer> lrPair 
                        = cart.getTree().get(currentIdx);

                if (pattern.getEntry(decisionPair.getKey())
                        > decisionPair.getValue()) {
                    // Left
                    currentIdx = lrPair.getKey();
                } else {
                    //Right
                    currentIdx = lrPair.getValue();
                }

            } else {
                postProb = cart.getPostProbTerminal().get(currentIdx);
                // Terminal Node Found 
                break;
            }
        }

        return postProb;
    }

    /**
     * Find the terminal node number from the CART TREE
     *
     * @param cart
     * @param pattern
     * @return
     */
    public static int findTerminalNodeNumber(CART cart,
            RealVector pattern) {

        // Starting IDX
        Integer currentIdx = 1;

        /**
         * note this loop is descending a finite tree (CART) and therefore
         * cannot be infinite (will reach a terminal node)
         */
        while (true) {
            if (cart.getDecisionNodes().containsKey(currentIdx)) {
                // Is a Decision

                ImmutablePair<Integer, Double> decisionPair
                        = cart.getDecisionNodes().get(currentIdx);

                ImmutablePair<Integer, Integer> lrPair 
                        = cart.getTree().get(currentIdx);

                if (pattern.getEntry(decisionPair.getKey())
                        > decisionPair.getValue()) {
                    // Left
                    currentIdx = lrPair.getKey();
                } else {
                    //Right
                    currentIdx = lrPair.getValue();
                }

            } else {
                // Is the terminal node, output idx
                return currentIdx;
            }
        }
    }

    //======================================================================
    /**
     * Get the minimum node size.
     *
     * @return
     */
    public int getMinNodeSize() {
        return minNodeSize;
    }

    /**
     * Get the alpha.
     *
     * @return
     */
    public double getAlpha() {
        return alpha;
    }

    /**
     * Get the minimum node size.
     *
     * @param minNode
     */
    public void setMinNodeSize(int minNode) {
        this.minNodeSize = minNode;
    }

    /**
     * Set the alpha.
     *
     * @param alpha
     */
    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    /**
     * Get the impurity type.
     *
     * @return
     */
    public ImpurityType getImpurityType() {
        return impurityType;
    }

    /**
     * Set the impurity type.
     *
     * @param impurityType
     */
    public void setImpurityType(ImpurityType impurityType) {
        this.impurityType = impurityType;
    }

}
