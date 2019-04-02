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
package fit.astro.vsa.utilities.ml.oc;

import fit.astro.vsa.common.utilities.math.linearalgebra.VectorOperations;
import fit.astro.vsa.common.utilities.test.classification.GrabIrisData;
import fit.astro.vsa.common.bindings.ml.ClassificationResult;
import fit.astro.vsa.common.datahandling.training.TrainCrossData;
import fit.astro.vsa.common.datahandling.training.TrainCrossTestGenerator;
import fit.astro.vsa.common.datahandling.LabelHandling;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class OneClassPWCAnalysis {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(OneClassPWCAnalysis.class);
    
    private static final Random RAND = new Random(42L);
    
    public static void main(String[] args) throws IOException {
        
        
        GrabIrisData grabIrisData = new GrabIrisData();
        Map<Integer, RealVector> setOfPatterns = grabIrisData.getSetOfPatterns();
        Map<Integer, String> setOfClasses = grabIrisData.getSetOfClasses();

        //=============================================================
        TrainCrossTestGenerator trainTest
                = new TrainCrossTestGenerator(
                        setOfClasses, 0.2, RAND);

        Map<Integer, List<Integer>> crossvalMap = trainTest.getCrossvalMap();
        // ==================================================================
        Map<String, Integer> unique = LabelHandling.countUniqueClasses(setOfClasses);
        String[] uniqueLabels = unique.keySet().toArray(new String[unique.keySet().size()]);

        double[] radii = VectorOperations.linearSpace(0.3, 0.005, 0.005);

        for (int kdx = 0; kdx < radii.length; kdx++) {

            double error = 0;

            for (Integer idx : crossvalMap.keySet()) {

                TrainCrossData crossDataPWC = new TrainCrossData(
                        setOfPatterns, setOfClasses, crossvalMap, idx);

                Map<Integer, String> reducedTrainingClasses
                        = crossDataPWC.getSetOfTrainingClasses().entrySet().stream()
                        .filter(p -> p.getValue().equals(uniqueLabels[0]) || p.getValue().equals(uniqueLabels[1]))
                        .collect(Collectors.toMap(p -> p.getKey(), p -> p.getValue()));

                Map<Integer, RealVector> reducedTrainingData = new HashMap<>(crossDataPWC.getSetOfTrainingPatterns());
                reducedTrainingData.keySet().retainAll(reducedTrainingClasses.keySet());

                // =============== Train and Apply Classifiers
                OneClassPWC pwc = new OneClassPWC(reducedTrainingData, "Known", "Anomaly");

                double threshold = pwc.train(radii[kdx]);
                ClassificationResult pwcWithoutResults = pwc.execute(radii[kdx],
                        crossDataPWC.getSetOfCrossvalPatterns(), threshold);

                Map<Integer, String> classEstimates
                        = pwcWithoutResults.getLabelEstimate();

                // ============= Estimate Error ========================
                double counter = 0;

                for (Integer jdx : crossDataPWC.getSetOfCrossvalPatterns().keySet()) {

                    if (classEstimates.get(jdx).equalsIgnoreCase("Anomaly")
                            && !crossDataPWC.getSetOfCrossvalClasses()
                            .get(jdx).equals(uniqueLabels[2])) {
                        counter = counter + 1;
                    }

                }

                double errorPart = counter
                        / (double) crossDataPWC.getSetOfCrossvalPatterns().size();

                error = error + errorPart;
            }

            double errorMean = error / 5.0;
            
            
            LOGGER.info("===========================================");
            LOGGER.info("With OC-PWC");
            LOGGER.info("Error: " + errorMean + "  with radius: " + radii[kdx]);

        }

    }

}
