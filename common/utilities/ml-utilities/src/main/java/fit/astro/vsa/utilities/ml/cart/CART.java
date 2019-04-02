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

import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import org.apache.commons.lang3.tuple.ImmutablePair;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class CART implements Serializable {

    // <editor-fold defaultstate="collapsed" desc="Variables">
    private static final long serialVersionUID = -6107478754148091667L;

    // Map of  Parent Index and the Pair of Children (L/R)
    private final Map<Integer, ImmutablePair<Integer, Integer>> tree;

    //Map of Index and The Components in the Node
    private final Map<Integer, HashSet<Integer>> nodes;

    //Map of Index to Dimension/Threshold Pair Used for Decisions
    private final Map<Integer, ImmutablePair<Integer, Double>> decisionNodes;

    //Map of Index to Parent Index
    private final Map<Integer, Integer> parentNodes;

    //Map of Index to Terminal Nodes
    private final Map<Integer, String> terminalNodes;
    private final Map<Integer, Double> terminalNodeImpurity;

    private final Map<Integer, Map<String, Double>> postProbTerminal;

    //List Of Unique Possible Labels;
    private final Map<String, Integer> uniqueLabelCount;

    // </editor-fold>
    
    /**
     * Make a CART object based on one already existing
     *
     * @param another
     */
    public CART(CART another) {
        this.tree = new HashMap<>(another.tree);
        this.nodes = new HashMap<>(another.nodes);
        this.decisionNodes = new HashMap<>(another.decisionNodes);
        this.parentNodes = new HashMap<>(another.parentNodes);
        this.terminalNodes = new HashMap<>(another.terminalNodes);
        this.terminalNodeImpurity = new HashMap<>(another.terminalNodeImpurity);
        this.uniqueLabelCount = new HashMap<>(another.uniqueLabelCount);
        this.postProbTerminal = new HashMap<>(another.postProbTerminal);
    }

    /**
     * Make the CART Object
     * 
     * @param tree
     * @param nodes
     * @param decisionNodes
     * @param parentNodes
     * @param terminalNodes
     * @param terminalNodeImpurity
     * @param uniqueLabelCount
     * @param postProbTerminal
     */
    public CART(Map<Integer, ImmutablePair<Integer, Integer>> tree,
            Map<Integer, HashSet<Integer>> nodes,
            Map<Integer, ImmutablePair<Integer, Double>> decisionNodes,
            Map<Integer, Integer> parentNodes, Map<Integer, String> terminalNodes,
            Map<Integer, Double> terminalNodeImpurity,
            Map<String, Integer> uniqueLabelCount,
            Map<Integer, Map<String, Double>> postProbTerminal) {
        this.tree = tree;
        this.nodes = nodes;
        this.decisionNodes = decisionNodes;
        this.parentNodes = parentNodes;
        this.terminalNodes = terminalNodes;
        this.terminalNodeImpurity = terminalNodeImpurity;
        this.uniqueLabelCount = uniqueLabelCount;
        this.postProbTerminal = postProbTerminal;
    }

    // <editor-fold defaultstate="collapsed" desc="Getter and Setters">
    /**
     *
     * @return
     */
    public Map<Integer, ImmutablePair<Integer, Integer>> getTree() {
        return tree;
    }

    /**
     *
     * @return
     */
    public Map<Integer, HashSet<Integer>> getNodes() {
        return nodes;
    }

    /**
     *
     * @return
     */
    public Map<Integer, ImmutablePair<Integer, Double>> getDecisionNodes() {
        return decisionNodes;
    }

    /**
     *
     * @return
     */
    public Map<Integer, Integer> getParentNodes() {
        return parentNodes;
    }

    /**
     *
     * @return
     */
    public Map<Integer, String> getTerminalNodes() {
        return terminalNodes;
    }

    /**
     *
     * @return
     */
    public Map<Integer, Double> getTerminalNodeImpurity() {
        return terminalNodeImpurity;
    }

    /**
     *
     * @return
     */
    public Map<String, Integer> getUniqueLabelCount() {
        return uniqueLabelCount;
    }

    /**
     *
     * @return
     */
    public Map<Integer, Map<String, Double>> getPostProbTerminal() {
        return postProbTerminal;
    }
// </editor-fold>
}
