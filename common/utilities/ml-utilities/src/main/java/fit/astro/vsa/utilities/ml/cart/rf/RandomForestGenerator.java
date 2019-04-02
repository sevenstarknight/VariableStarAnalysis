/*
 * Copyright (C) 2019 kjohnston
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
package fit.astro.vsa.utilities.ml.cart.rf;

import fit.astro.vsa.common.bindings.ml.ClassificationResult;
import fit.astro.vsa.common.datahandling.LabelHandling;
import fit.astro.vsa.common.datahandling.training.TrainCrossData;
import fit.astro.vsa.common.datahandling.training.TrainCrossGenerator;
import fit.astro.vsa.common.utilities.math.support.SortingOperations;
import fit.astro.vsa.utilities.ml.cart.CART;
import fit.astro.vsa.utilities.ml.cart.CARTClassifierGenerator;
import fit.astro.vsa.utilities.ml.cart.ImpurityType;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author kjohnston
 */
public class RandomForestGenerator {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(RandomForestGenerator.class);
    
    // ============== Input
    private final Map<Integer, RealVector> setOfPatternsTraining;
    private final Map<Integer, String> setOfClassesTraining;
    private final int numTrees;
    // ============== Internal
    private int dimensions;

    private Random rand = new Random();
    private static String missedLabel = "missed";
    private ImpurityType impurityType = ImpurityType.GDI;
    private int minNodeSize = 5;
    private double alpha = 0.001;

    /**
     * Initialize CART Generator
     *
     * @param setOfPatternsTraining
     * @param setOfClassesTraining
     * @param numTrees
     */
    public RandomForestGenerator(
            Map<Integer, RealVector> setOfPatternsTraining,
            Map<Integer, String> setOfClassesTraining, int numTrees) {
        this.setOfPatternsTraining = setOfPatternsTraining;
        this.setOfClassesTraining = setOfClassesTraining;
        this.numTrees = numTrees;
    }

    public RandomForest generateRF(int sample, double alpha) {

        List<Integer> keys = new ArrayList<>(setOfPatternsTraining.keySet());

        Map<String, Integer> uniqueLabelCount
                = LabelHandling.countUniqueClasses(setOfClassesTraining);

        List<CART> rf = new ArrayList<>();

        for (int idx = 0; idx < numTrees; idx++) {
            
            // Subsect the tree
            Map<Integer, RealVector> setOfPatternsSub = new HashMap<>();
            Map<Integer, String> setOfClassesSub = new HashMap<>();

            List<Integer> keySet = new ArrayList<>(setOfPatternsTraining.keySet());
            
            // Note this is a map, so there are no replicate keys in the set
            for (int jdx = 0; jdx < sample; jdx++) {
                // Sampling WITH replacement
                Integer randKey = keySet.get(rand.nextInt(keySet.size()));
                setOfPatternsSub.put(randKey, setOfPatternsTraining.get(randKey));
                setOfClassesSub.put(randKey, setOfClassesTraining.get(randKey));
            }

            TrainCrossGenerator crossGenerator = new TrainCrossGenerator(setOfClassesSub, rand);

            TrainCrossData crossDataCART = new TrainCrossData(
                    setOfPatternsSub, setOfClassesSub, crossGenerator.getCrossvalMap(), 0);

            if (idx % 10 == 0) {
                LOGGER.info("Tree Number: " + idx);
            }
            
            // ==================================================================
            // =============== Train and Apply Classifiers
            CARTClassifierGenerator cartGenerator
                    = new CARTClassifierGenerator(
                            crossDataCART.getSetOfTrainingPatterns(),
                            crossDataCART.getSetOfTrainingClasses());

            CART cart = cartGenerator.generateCART(alpha,
                    crossDataCART.getSetOfCrossvalPatterns(),
                    crossDataCART.getSetOfCrossvalClasses());

            rf.add(cart);
        }

        return new RandomForest(rf, uniqueLabelCount);
    }

    //======================================================================
    // Use CART Object
    /**
     *
     * @param rf
     * @param inputPatternMap
     * <p>
     * @return
     */
    public static ClassificationResult execute(RandomForest rf,
            Map<Integer, RealVector> inputPatternMap) {

        Map<Integer, String> labelEstimate = new HashMap<>();
        Map<Integer, Map<String, Double>> labelPostProb = new HashMap<>();

        // ========================================        
        // Loop Over Cross-Validation Data
        for (Integer idx : inputPatternMap.keySet()) {

            RealVector crossvalPattern = inputPatternMap.get(idx);

            List<String> listOfLabels = new ArrayList<>();
            rf.getRf().forEach((cart) -> {
                listOfLabels.add(CARTClassifierGenerator.
                        estimateClassLabel(cart, crossvalPattern));
            });

            Map<String, Long> freq = listOfLabels.stream()
                    .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));

            Map<String, Long> sortedNeighbors
                    = SortingOperations.sortByDecendingValue(freq);

            List<Entry<String, Long>> knnList = new ArrayList<>(
                    sortedNeighbors.entrySet());

            labelEstimate.put(idx, knnList.get(0).getKey());
            

            Map<String, Double> probEst = new HashMap<>();
            for (String label : sortedNeighbors.keySet()) {
                double probEstTmp = (double) sortedNeighbors.get(label) / (double) rf.getRf().size();
                probEst.put(label, probEstTmp);
            }

            labelPostProb.put(idx, probEst);

        }

        return new ClassificationResult(labelEstimate, labelPostProb,
                rf.getUniqueLabelCount());
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

    public void setRand(Random rand) {
        this.rand = rand;
    }

}
