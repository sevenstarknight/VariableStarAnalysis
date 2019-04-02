/*
 * Copyright (C) 2018 kjohnston
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
package fit.astro.vsa.analysis.ucr;

import fit.astro.vsa.analysis.ProcessRawRF;
import fit.astro.vsa.analysis.df.generators.DFGenerator;
import fit.astro.vsa.analysis.df.generators.DFOptions;
import fit.astro.vsa.analysis.ssmm.generators.SSMMGenerator;
import fit.astro.vsa.common.bindings.ml.TimeDomainAttributeMaps;
import fit.astro.vsa.common.bindings.math.Real2DCurve;
import fit.astro.vsa.common.datahandling.training.TrainCrossGenerator;
import fit.astro.vsa.common.utilities.io.ReadingInUCRData;
import fit.astro.vsa.common.utilities.io.dataset.ReadingInLINEARData;
import fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException;
import fit.astro.vsa.common.utilities.math.linearalgebra.MatrixOperations;
import fit.astro.vsa.utilities.ml.training.NormalizeData;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.util.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author kjohnston
 */
public class RawRF_UCR_Vector extends ProcessRawRF {

    private static final Random RAND = new Random(42L);

    private static final Logger LOGGER
            = LoggerFactory.getLogger(RawRF_UCR_Vector.class);

    private static Pair<Map<String, RealVector>, Map<String, RealVector>> transformationVectors;

    public static void main(String[] args) throws IOException, NotEnoughDataException {

        String inputLocation = "UCR";
        // =================================================================

        double alpha = 0.01;
        if (args.length > 0) {
            try {
                alpha = Double.parseDouble(args[0]);
            } catch (NumberFormatException e) {
                System.err.println("Argument" + args[0] + " must be a int. Using 0.01 instead");
                alpha = 0.01;
            }
        }
        
        boolean isTest = Boolean.FALSE;
        if (args.length > 1) {
            String testOption = args[1];
            if (testOption.equalsIgnoreCase("-test")) {
                isTest = Boolean.TRUE;
            }
        }

        try {
            execute(alpha, isTest, inputLocation);
        } catch (IOException | NotEnoughDataException ex) {
            LOGGER.warn(ex.getMessage());
        }
    }

    /**
     *
     * @param alpha
     * @param isTest
     * @param location
     * @throws IOException
     * @throws NotEnoughDataException
     */
    public static void execute(double alpha, boolean isTest, String location) 
            throws IOException, NotEnoughDataException {

        // Pull Data
        ReadingInUCRData starLight = new ReadingInUCRData(location);

        TimeDomainAttributeMaps trainingData = starLight.getTrainData("StarLightCurves");
        TimeDomainAttributeMaps testingData = starLight.getTestData("StarLightCurves");

        generatePatterns(ReadingInLINEARData.Views.Statistics.toString(), trainingData, testingData);

        if (!isTest) {
            trainRF("UCR" + ReadingInLINEARData.Views.Statistics.toString());
        } else {
            testRF(alpha, "UCR" + ReadingInLINEARData.Views.Statistics.toString());

        }

        // =================================================================
        generatePatterns(ReadingInLINEARData.Views.DF.toString(), trainingData, testingData);

        if (!isTest) {
            trainRF("UCR" + ReadingInLINEARData.Views.DF.toString());
        } else {
            testRF(alpha, "UCR" + ReadingInLINEARData.Views.DF.toString());

        }

        // =================================================================
        generatePatterns(ReadingInLINEARData.Views.SSMM.toString(), trainingData, testingData);

        if (!isTest) {
            trainRF("UCR" + ReadingInLINEARData.Views.SSMM.toString());
        } else {
            testRF(alpha, "UCR" + ReadingInLINEARData.Views.SSMM.toString());

        }

    }

    private static void generatePatterns(String view, TimeDomainAttributeMaps trainingData,
            TimeDomainAttributeMaps testingData) {
        // =======================================================
        // Feature Processing
        DFOptions dFOptions = new DFOptions(30, 25,
                new int[]{7, 1}, 0.4, DFOptions.Directions.both);

        DFGenerator dfGenerator
                = new DFGenerator(dFOptions);

        double windowWidth = SSMMGenerator.estimateWindowWidth(
                trainingData.getSetOfWaveforms().values().iterator().next(), 35);

        SSMMGenerator ssmmGenerator
                = new SSMMGenerator(0.06, windowWidth);

        TrainCrossGenerator trainTest
                = new TrainCrossGenerator(trainingData.getSetOfClasses(), RAND);

        crossvalMap = trainTest.getCrossvalMap();

        Map<Integer, Map<String, RealVector>> setOfPatterns_Train = new HashMap<>(
                trainingData.getSetOfClasses().keySet().size());
        
        setOfClasses_Training = new HashMap<>();

        for (Integer idx : trainingData.getSetOfClasses().keySet()) {

            Map<String, RealVector> pattern = new HashMap<>();

            Real2DCurve currentWaveform = trainingData.getSetOfWaveforms().get(idx);

            // statistics are on the amplitude of the waveform so phasing doesn't matter
            DescriptiveStatistics descript = new DescriptiveStatistics(currentWaveform
                    .getYArrayPrimitive());

            // input waveform is already phased for UCR data
            RealMatrix df = dfGenerator.evaluate(currentWaveform);

            pattern.put(ReadingInLINEARData.Views.DF.toString(), new ArrayRealVector(
                    MatrixOperations.unpackMatrix(df)));

            // input waveform doesn't require phasing
            RealMatrix ssmm = ssmmGenerator.evaluate(currentWaveform);

            pattern.put(ReadingInLINEARData.Views.SSMM.toString(), new ArrayRealVector(
                    MatrixOperations.unpackMatrix(ssmm)));

            RealVector statistics = new ArrayRealVector(
                    new double[]{descript.getMean(), descript.getStandardDeviation(),
                        descript.getSkewness(), descript.getKurtosis()});

            pattern.put(ReadingInLINEARData.Views.Statistics.toString(), statistics);

            setOfPatterns_Train.put(idx, pattern);
            setOfClasses_Training.put(idx, trainingData.getSetOfClasses().get(idx));

        }

        // ======================================================
        // Training Data
        transformationVectors = NormalizeData.normalizeMultiViewVectorVariate(setOfPatterns_Train);

        Map<Integer, Map<String, RealVector>> setOfPatterns_TrainTmp = 
                NormalizeData.applyNormalizeVectorVariate(setOfPatterns_Train,
                transformationVectors);

        setOfPatterns_Training = new HashMap<>();
        setOfPatterns_TrainTmp.keySet().forEach((idx) -> {
            setOfPatterns_Training.put(idx,
                    setOfPatterns_TrainTmp.get(idx).get(view));
        });

        // ======================================================
        //  Testing Data
        Map<Integer, Map<String, RealVector>> setOfPatterns_Test = new HashMap<>(
                testingData.getSetOfClasses().keySet().size());

        setOfClasses_Testing = new HashMap<>();
        
        for (Integer idx : testingData.getSetOfClasses().keySet()) {

            Map<String, RealVector> pattern = new HashMap<>();

            Real2DCurve currentWaveform = testingData.getSetOfWaveforms().get(idx);

            // statistics are on the amplitude of the waveform so phasing doesn't matter
            DescriptiveStatistics descript = new DescriptiveStatistics(currentWaveform
                    .getYArrayPrimitive());

            // input waveform is already phased for UCR data
            RealMatrix df = dfGenerator.evaluate(currentWaveform);

            pattern.put(ReadingInLINEARData.Views.DF.toString(), new ArrayRealVector(
                    MatrixOperations.unpackMatrix(df)));

            // input waveform doesn't require phasing
            RealMatrix ssmm = ssmmGenerator.evaluate(currentWaveform);

            pattern.put(ReadingInLINEARData.Views.SSMM.toString(), new ArrayRealVector(
                    MatrixOperations.unpackMatrix(ssmm)));

            RealVector statistics = new ArrayRealVector(
                    new double[]{descript.getMean(), descript.getStandardDeviation(),
                        descript.getSkewness(), descript.getKurtosis()});

            pattern.put(ReadingInLINEARData.Views.Statistics.toString(), statistics);

            setOfPatterns_Test.put(idx, pattern);
            setOfClasses_Testing.put(idx, testingData.getSetOfClasses().get(idx));

        }

        Map<Integer, Map<String, RealVector>> setOfPatterns_TestTmp = 
                NormalizeData.applyNormalizeVectorVariate(setOfPatterns_Test,
                transformationVectors);

        setOfPatterns_Testing = new HashMap<>();
        setOfPatterns_TestTmp.keySet().forEach((idx) -> {
            setOfPatterns_Testing.put(idx,
                    setOfPatterns_TestTmp.get(idx).get(view));
        });

    }
}
