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
package fit.astro.vsa.utilities.ml.ldaqda;

import fit.astro.vsa.common.bindings.ml.DiscrimantAnalysisMethod;
import fit.astro.vsa.common.utilities.test.classification.GrabIrisData;
import fit.astro.vsa.common.bindings.ml.ClassificationResult;
import fit.astro.vsa.utilities.ml.performance.ClassifierPerformance;
import fit.astro.vsa.common.datahandling.training.TrainCrossData;
import fit.astro.vsa.common.datahandling.training.TrainCrossTestGenerator;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.commons.math3.linear.RealVector;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author SevenStarKnight
 */
public class LDAQDATest {

    private final Random RAND = new Random(42L);

    
    private static final Logger LOGGER
            = LoggerFactory.getLogger(LDAQDATest.class);
    
    public LDAQDATest() {
    }

    private Map<Integer, RealVector> setOfPatterns;
    private Map<Integer, String> setOfClasses;

    private Map<Integer, List<Integer>> crossvalMap;


    @Before
    public void setUp() throws IOException, URISyntaxException {

        GrabIrisData grabIrisData = new GrabIrisData();
        this.setOfPatterns = grabIrisData.getSetOfPatterns();
        this.setOfClasses = grabIrisData.getSetOfClasses();

        //=============================================================
        TrainCrossTestGenerator trainTest
                = new TrainCrossTestGenerator(
                        setOfClasses, 0.2, RAND);

        crossvalMap = trainTest.getCrossvalMap();

    }

    @Test
    public void testLDAQDA_LDAGeneral() {

        // ==================================================================
        double withError = 0;

        for (Integer idx : crossvalMap.keySet()) {

            TrainCrossData crossData = new TrainCrossData(
                    setOfPatterns, setOfClasses, crossvalMap, idx);

            // =============== Train and Apply Classifiers
            LDAQDAClassifier ldaqda = new LDAQDAClassifier(
                    crossData.getSetOfTrainingPatterns(),
                    crossData.getSetOfTrainingClasses());

            DiscriminantAnalysis analysis
                    = ldaqda.generateDiscriminateAnalysis(
                            DiscrimantAnalysisMethod.LDA_GENERAL);

            ClassificationResult classificationResult
                    = ldaqda.execute(analysis, crossData.getSetOfCrossvalPatterns());

            double errorCanon = ClassifierPerformance.
                    estimateMisclassificationError(
                            classificationResult, crossData.getSetOfCrossvalClasses());

            withError += errorCanon / (double) crossData.getSetOfCrossvalClasses().size();

        }

        
        LOGGER.info("LDA General: " + withError);
    }

    @Test
    public void testLDAQDA_LDAISO() {

        // ==================================================================
        double withError = 0;

        for (Integer idx : crossvalMap.keySet()) {

            TrainCrossData crossData = new TrainCrossData(
                    setOfPatterns, setOfClasses, crossvalMap, idx);

            // =============== Train and Apply Classifiers
            LDAQDAClassifier ldaqda = new LDAQDAClassifier(
                    crossData.getSetOfTrainingPatterns(),
                    crossData.getSetOfTrainingClasses());

            DiscriminantAnalysis analysis
                    = ldaqda.generateDiscriminateAnalysis(
                            DiscrimantAnalysisMethod.LDA_ISOTROPIC);

            ClassificationResult classificationResult
                    = ldaqda.execute(analysis,
                            crossData.getSetOfCrossvalPatterns());

            double errorCanon = ClassifierPerformance.
                    estimateMisclassificationError(
                            classificationResult,
                            crossData.getSetOfCrossvalClasses());

            withError += errorCanon / (double) crossData.getSetOfCrossvalClasses().size();

        }

        LOGGER.info("LDA Iso: " + withError);
    }

    @Test
    public void testLDAQDA_LDANaive() {

        // ==================================================================
        double withError = 0;

        for (Integer idx : crossvalMap.keySet()) {

            TrainCrossData crossData = new TrainCrossData(
                    setOfPatterns, setOfClasses, crossvalMap, idx);

            // =============== Train and Apply Classifiers
            LDAQDAClassifier ldaqda = new LDAQDAClassifier(
                    crossData.getSetOfTrainingPatterns(),
                    crossData.getSetOfTrainingClasses());

            DiscriminantAnalysis analysis
                    = ldaqda.generateDiscriminateAnalysis(
                            DiscrimantAnalysisMethod.LDA_NAIVE);

            ClassificationResult classificationResult
                    = ldaqda.execute(analysis, crossData.getSetOfCrossvalPatterns());

            double errorCanon = ClassifierPerformance.
                    estimateMisclassificationError(
                            classificationResult, crossData.getSetOfCrossvalClasses());

            withError += errorCanon / (double) crossData.getSetOfCrossvalClasses().size();

        }

        LOGGER.info("LDA Naive: " + withError);
    }

    @Test
    public void testLDAQDA_QDANaive() {

        // ==================================================================
        double withError = 0;

        for (Integer idx : crossvalMap.keySet()) {

            TrainCrossData crossData = new TrainCrossData(
                    setOfPatterns, setOfClasses, crossvalMap, idx);

            // =============== Train and Apply Classifiers
            LDAQDAClassifier ldaqda = new LDAQDAClassifier(
                    crossData.getSetOfTrainingPatterns(),
                    crossData.getSetOfTrainingClasses());

            DiscriminantAnalysis analysis
                    = ldaqda.generateDiscriminateAnalysis(
                            DiscrimantAnalysisMethod.QDA_NAIVE);

            ClassificationResult classificationResult
                    = ldaqda.execute(analysis, crossData.getSetOfCrossvalPatterns());

            double errorCanon = ClassifierPerformance.
                    estimateMisclassificationError(
                            classificationResult, crossData.getSetOfCrossvalClasses());

            withError += errorCanon / (double) crossData.getSetOfCrossvalClasses().size();

        }

        LOGGER.info("QDA Naive: " + withError);
    }

    @Test
    public void testLDAQDA_QDAIso() {

        // ==================================================================
        double withError = 0;

        for (Integer idx : crossvalMap.keySet()) {

            TrainCrossData crossData = new TrainCrossData(
                    setOfPatterns, setOfClasses, crossvalMap, idx);

            // =============== Train and Apply Classifiers
            LDAQDAClassifier ldaqda = new LDAQDAClassifier(
                    crossData.getSetOfTrainingPatterns(),
                    crossData.getSetOfTrainingClasses());

            DiscriminantAnalysis analysis
                    = ldaqda.generateDiscriminateAnalysis(
                            DiscrimantAnalysisMethod.QDA_ISOTROPIC);

            ClassificationResult classificationResult
                    = ldaqda.execute(analysis, crossData.getSetOfCrossvalPatterns());

            double errorCanon = ClassifierPerformance.
                    estimateMisclassificationError(
                            classificationResult, crossData.getSetOfCrossvalClasses());

            withError += errorCanon / (double) crossData.getSetOfCrossvalClasses().size();

        }

        LOGGER.info("QDA Iso: " + withError);
    }

    @Test
    public void testLDAQDA_QDAGeneral() {

        // ==================================================================
        double withError = 0;

        for (Integer idx : crossvalMap.keySet()) {

            TrainCrossData crossData = new TrainCrossData(
                    setOfPatterns, setOfClasses, crossvalMap, idx);

            // =============== Train and Apply Classifiers
            LDAQDAClassifier ldaqda = new LDAQDAClassifier(
                    crossData.getSetOfTrainingPatterns(),
                    crossData.getSetOfTrainingClasses());

            DiscriminantAnalysis analysis
                    = ldaqda.generateDiscriminateAnalysis(
                            DiscrimantAnalysisMethod.QDA_GENERAL);

            ClassificationResult classificationResult
                    = ldaqda.execute(analysis, crossData.getSetOfCrossvalPatterns());

            double errorCanon = ClassifierPerformance.
                    estimateMisclassificationError(
                            classificationResult, crossData.getSetOfCrossvalClasses());

            withError += errorCanon / (double) crossData.getSetOfCrossvalClasses().size();

        }

        LOGGER.info("QDA General: " + withError);
    }

}
