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

import fit.astro.vsa.common.utilities.math.support.SortingOperations;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;
import org.apache.commons.lang3.tuple.ImmutablePair;
import static fit.astro.vsa.utilities.ml.cart.CARTClassifierGenerator.estimateMisclassificationRate;
import org.apache.commons.math3.linear.RealVector;
import fit.astro.vsa.common.datahandling.LabelHandling;


/**
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class CARTPruneAndGrow {

    // ============================================
    // Input
    private final Map<Integer, RealVector> setOfTrainingData;
    private final Map<Integer, String> setOfTrainingClasses;
    private final int dimensions;
    private final ImpurityType impurityType;

    // ============================================
    // Internal
    private final Map<String, Integer> uniqueLabelCount;
    
    private double TOL = 0.001;
    private int minNodeSize = 3;

    public CARTPruneAndGrow(Map<Integer, RealVector> setOfTrainingData,
            Map<Integer, String> setOfTrainingClasses,
            int dimensions, ImpurityType impurityType) {
        this.setOfTrainingData = setOfTrainingData;
        this.setOfTrainingClasses = setOfTrainingClasses;
        this.dimensions = dimensions;
        this.impurityType = impurityType;
        this.uniqueLabelCount
                = LabelHandling.countUniqueClasses(setOfTrainingClasses);
    }

    /**
     * Bottom up pruning
     * <p>
     * @param alpha
     * @param cart
     * @param inputPatternMap
     * @param inputClassMap
     * <p>
     * @return
     */
    public CART PruneCART(double alpha, CART cart,
            Map<Integer, RealVector> inputPatternMap,
            Map<Integer, String> inputClassMap) {
        // ================================================================
        // Run CART, Estimate Misclassification

        double errorFullTree
                = estimateMisclassificationRate(cart, inputPatternMap, inputClassMap);

        int numNodes = cart.getTerminalNodes().size();

        // ================================================================
        // Prune Tree     
        Boolean changeWasMade = Boolean.TRUE;
        while (changeWasMade) {

            Set<Integer> newSet = new TreeSet<>(cart.getTerminalNodes().keySet());
            changeWasMade = Boolean.FALSE;

            // Cycle Over Terminal Nodes
            for (Integer idx : newSet) {

                CART tmpCart = new CART(cart);

                /**
                 * We've disconnected the for loop from the set of terminal
                 * nodes, now we need to check if we've removed that node
                 * already (the right node in a pair, mostly)
                 */
                if (!tmpCart.getTerminalNodes().containsKey(idx)) {
                    continue;
                }

                // For the Current Terminal Node, Get the Parent
                Integer parentID = tmpCart.getParentNodes().get(idx);

                // Find the Sibling
                ImmutablePair<Integer, Integer> lrPair = tmpCart.getTree().get(parentID);

                //Are the Siblings Terminal Nodes?
                if (lrPair == null) {
                    tmpCart.getTree();
                }

                if (!(tmpCart.getTerminalNodes().containsKey(lrPair.getKey())
                        && tmpCart.getTerminalNodes().containsKey(lrPair.getValue()))) {
                    // They are not, go to the next terminal node
                    continue;
                }

                //If they are, then prune
                //=============================================================
                // Remove Children
                // Remove the children from the terminal node map
                tmpCart.getTerminalNodes().remove(lrPair.getKey());
                tmpCart.getTerminalNodes().remove(lrPair.getValue());

                tmpCart.getTerminalNodeImpurity().remove(lrPair.getKey());
                tmpCart.getTerminalNodeImpurity().remove(lrPair.getValue());

                tmpCart.getPostProbTerminal().remove(lrPair.getKey());
                tmpCart.getPostProbTerminal().remove(lrPair.getValue());

                // Remove the children from the Parent node map
                tmpCart.getParentNodes().remove(lrPair.getKey());
                tmpCart.getParentNodes().remove(lrPair.getValue());

                //Remove the children from the nodes map
                tmpCart.getNodes().remove(lrPair.getKey());
                tmpCart.getNodes().remove(lrPair.getValue());

                //=============================================================
                // Parent is no longer a decision (but is everything else)
                tmpCart.getDecisionNodes().remove(parentID);

                tmpCart.getTree().remove(parentID);
                // =============================================================
                // New Terminal
                Map<Integer, RealVector> setOfData = new HashMap<>();
                Map<Integer, String> setOfClasses = new HashMap<>();

                if (!tmpCart.getNodes().containsKey(parentID)) {
                    tmpCart.getNodes().containsKey(parentID);
                }

                // Map Based on Current Index Values
                for (Integer jdx : tmpCart.getNodes().get(parentID)) {
                    setOfData.put(jdx, setOfTrainingData.get(jdx));
                    setOfClasses.put(jdx, setOfTrainingClasses.get(jdx));
                }

                // ===========================================================
                // Store Terminal Node
                Map<String, Map<Integer, RealVector>> nodeMembers
                        = LabelHandling.sortIntoMaps(setOfData, setOfClasses);

                // Post Probs
                Map<String, Double> postProb = new HashMap<>();
                int count = 0;
                count = nodeMembers.keySet().stream().map((e)
                        -> nodeMembers.get(e).size()).reduce(count, Integer::sum);

                for (String e : nodeMembers.keySet()) {
                    postProb.put(e,
                            (double) nodeMembers.get(e).size() / (double) count);
                }

                // Label
                String classLabel;
                if (nodeMembers.keySet().size() == 1) {
                    classLabel = nodeMembers.keySet().iterator().next();
                } else {

                    // Sort Lengths
                    Map<String, Integer> tmpMapLengths = new HashMap<>();
                    nodeMembers.keySet().stream().forEach((e) -> {
                        tmpMapLengths.put(e, nodeMembers.get(e).size());
                    });

                    Map<String, Integer> sortedDistances
                            = SortingOperations.sortByAcendingValue(tmpMapLengths);

                    // Get top two
                    Iterator iterLabels = sortedDistances.keySet().iterator();
                    String primeLabel = (String) iterLabels.next();
                    Integer first = sortedDistances.get(primeLabel);
                    Integer second = sortedDistances.get((String) iterLabels.next());

                    //If the top two aren't equal, we have a winner
                    if (!Objects.equals(first, second)) {
                        classLabel = primeLabel;
                    } else {
                        classLabel = "Missed";
                    }

                }

                double impurity = FindSplit.EstimateImpurity(impurityType, setOfClasses);

                tmpCart.getTerminalNodes().put(parentID, classLabel);
                tmpCart.getPostProbTerminal().put(parentID, postProb);
                tmpCart.getTerminalNodeImpurity().put(parentID, impurity);

                tmpCart.getNodes().remove(lrPair.getKey());
                tmpCart.getNodes().remove(lrPair.getValue());

                //============================================================
                // Estimate the cost of pruning
                double errorPrunedTree
                        = estimateMisclassificationRate(tmpCart,
                                inputPatternMap, inputClassMap);

                double g_star = (errorPrunedTree - errorFullTree)
                        / ((double) numNodes - (double) tmpCart.getTerminalNodes().size());

                //Keep the Pruned TREE;
                if (g_star < alpha) {
                    cart = new CART(tmpCart);
                    changeWasMade = Boolean.TRUE;
                }

            }

        }

        return cart;
    }

    /**
     * Top down growth, left to right
     * <p>
     * @param impurityType
     * <p>
     * @return
     */
    public CART GrowCART(ImpurityType impurityType) {

        //=========================================================
        // Grow Tree
        // Map of  Parent Index and the Pair of Children (L/R)
        Map<Integer, ImmutablePair<Integer, Integer>> tree = new HashMap<>();

        //Map of Index and The Components in the Node
        Map<Integer, HashSet<Integer>> nodes = new HashMap<>();
        //Map of Index to Dimension/Threshold Pair Used for Decisions
        Map<Integer, ImmutablePair<Integer, Double>> decisionNodes = new HashMap<>();
        //Map of Index to Parent Index
        Map<Integer, Integer> parentNodes = new HashMap<>();
        //Map of Index to Terminal Nodes
        Map<Integer, String> terminalNodes = new HashMap<>();
        Map<Integer, Double> terminalNodeImpurity = new HashMap<>();

        Map<Integer, Map<String, Double>> postProbTerminal = new HashMap<>();

        //=========================================================
        Integer currentIdx = 1;
        Integer maxIdx = 1;
        nodes.put(currentIdx, new HashSet<>(setOfTrainingData.keySet()));

        Boolean growingTree = Boolean.TRUE;

        //Construct Tree
        while (growingTree) {

            // Parent Node Storage
            Map<Integer, RealVector> setOfData = new HashMap<>();
            Map<Integer, String> setOfClasses = new HashMap<>();

            // Map Based on Current Index Values
            for (Integer idx : nodes.get(currentIdx)) {
                setOfData.put(idx, setOfTrainingData.get(idx));
                setOfClasses.put(idx, setOfTrainingClasses.get(idx));
            }

            double impurityParent = FindSplit.EstimateImpurity(
                    impurityType, setOfClasses);

            ImmutablePair<Integer, Double> decisionPair
                    = FindSplit.GenerateSplit(impurityType, setOfData,
                            setOfClasses, dimensions);

            // Impurity Threshold and Data Size Threshold
            if (impurityParent > TOL
                    && setOfData.size() > minNodeSize
                    && decisionPair.getLeft() != null) {
                //MAKE A DECISION NODE
                // ============================================================

                //Put in storage of the child nodes 
                tree.put(currentIdx, new ImmutablePair<>(maxIdx + 1, maxIdx + 2));
                decisionNodes.put(currentIdx, decisionPair);

                // Split Data
                Map<Integer, String> setOfLeftClasses = new HashMap<>();
                Map<Integer, String> setOfRightClasses = new HashMap<>();

                for (Integer idx : setOfData.keySet()) {
                    if (setOfData.get(idx).getEntry(decisionPair.getKey())
                            > decisionPair.getValue()) {
                        setOfLeftClasses.put(idx, setOfClasses.get(idx));
                    } else {
                        setOfRightClasses.put(idx, setOfClasses.get(idx));
                    }
                }

                // Store New Nodes
                nodes.put(maxIdx + 1, new HashSet<>(setOfLeftClasses.keySet()));
                nodes.put(maxIdx + 2, new HashSet<>(setOfRightClasses.keySet()));

                parentNodes.put(maxIdx + 1, currentIdx);
                parentNodes.put(maxIdx + 2, currentIdx);

                // Where you are now
                currentIdx = maxIdx + 1;
                // What is the MAX IDX After Split
                maxIdx = maxIdx + 2;
            } else {
                // Is A Terminal Node, Move Up The Tree To Find the Next Node

                // Store Terminal Node
                Map<String, Map<Integer, RealVector>> nodeMembers
                        = LabelHandling.sortIntoMaps(setOfData, setOfClasses);

                // Post Probs
                Map<String, Double> postProb = new HashMap<>();
                int count = 0;
                count = nodeMembers.keySet().stream().map((e)
                        -> nodeMembers.get(e).size()).reduce(count, Integer::sum);

                for (String e : nodeMembers.keySet()) {
                    postProb.put(e,
                            (double) nodeMembers.get(e).size() / (double) count);
                }

                double impurity = FindSplit.EstimateImpurity(impurityType, setOfClasses);

                // Label
                String classLabel;
                if (nodeMembers.keySet().size() == 1) {
                    classLabel = nodeMembers.keySet().iterator().next();
                } else {

                    // Sort Lengths
                    Map<String, Integer> tmpMapLengths = new HashMap<>();
                    nodeMembers.keySet().stream().forEach((e) -> {
                        tmpMapLengths.put(e, nodeMembers.get(e).size());
                    });

                    Map<String, Integer> sortedDistances
                            = SortingOperations.sortByAcendingValue(tmpMapLengths);

                    // Get top two
                    Iterator iterLabels = sortedDistances.keySet().iterator();
                    String primeLabel = (String) iterLabels.next();
                    Integer first = sortedDistances.get(primeLabel);
                    Integer second = sortedDistances.get((String) iterLabels.next());

                    //If the top two aren't equal, we have a winner
                    if (!Objects.equals(first, second)) {
                        classLabel = primeLabel;
                    } else {
                        classLabel = "Missed";
                    }

                }
                terminalNodes.put(currentIdx, classLabel);
                postProbTerminal.put(currentIdx, postProb);
                terminalNodeImpurity.put(currentIdx, impurity);

                // ================== Make your next move ====================
                while (Boolean.TRUE) {

                    if (parentNodes.containsKey(currentIdx)) {
                        Integer parentIdx = parentNodes.get(currentIdx);
                        ImmutablePair<Integer, Integer> children = tree.get(parentIdx);

                        if (Objects.equals(children.getKey(), currentIdx)) {
                            //Move to the Right Node
                            currentIdx = children.getValue();
                            break;
                        } else {
                            // this the right node
                            currentIdx = parentIdx;
                        }
                    } else {
                        //is the root, therefore quit!!
                        growingTree = Boolean.FALSE;
                        break;
                    }

                }

            }

        }

        return new CART(tree, nodes, decisionNodes,
                parentNodes,
                terminalNodes, terminalNodeImpurity,
                uniqueLabelCount, postProbTerminal);
    }

    public void setTOL(double TOL) {
        this.TOL = TOL;
    }

    public void setMinNodeSize(int minNodeSize) {
        this.minNodeSize = minNodeSize;
    }

    public double getTOL() {
        return TOL;
    }

    public int getMinNodeSize() {
        return minNodeSize;
    }

}
