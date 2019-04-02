/*
 * Copyright (C) 2018 Kyle Johnston <kyjohnst2000@my.fit.edu>
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

import fit.astro.vsa.analysis.ProcessTimeDomainViaL3ML_MV;
import fit.astro.vsa.analysis.df.generators.DFGenerator;
import fit.astro.vsa.analysis.df.generators.DFOptions;
import fit.astro.vsa.analysis.ssmm.generators.SSMMGenerator;
import fit.astro.vsa.common.bindings.ml.TimeDomainAttributeMaps;
import fit.astro.vsa.common.bindings.math.Real2DCurve;
import fit.astro.vsa.common.datahandling.training.TrainCrossGenerator;
import fit.astro.vsa.common.utilities.io.ReadingInUCRData;
import fit.astro.vsa.common.utilities.io.dataset.ReadingInLINEARData;
import fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException;
import fit.astro.vsa.utilities.ml.training.NormalizeData;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.util.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class ProcessUCRDataViaL3ML_MV extends ProcessTimeDomainViaL3ML_MV {

    private static final Random RAND = new Random(42L);

    private static final Logger LOGGER
            = LoggerFactory.getLogger(ProcessUCRDataViaL3ML_MV.class);

    // =======================================================
    private static Pair<Map<String, RealMatrix>, Map<String, RealMatrix>> transformationVectors;

    /**
     *
     * @param args [lambda, mu, gamma] a fourth option can be provided "-test"
     * that will have the algorithm run in test mode.
     */
    public static void main(String[] args) {

        String inputLocation = "UCR";
        // =================================================================
        double lambda = 0.5;
        if (args.length > 0) {
            try {
                lambda = Double.parseDouble(args[0]);
            } catch (NumberFormatException e) {
                System.err.println("Argument" + args[0] + " must be a double. Using 0.5 instead");
                lambda = 0.5;
            }
        }

        double mu = 0.5;
        if (args.length > 1) {
            try {
                mu = Double.parseDouble(args[1]);
            } catch (NumberFormatException e) {
                System.err.println("Argument" + args[1] + " must be a double. Using 0.5 instead");
                mu = 0.5;
            }
        }

        double gamma = 0.5;
        if (args.length > 2) {
            try {
                gamma = Double.parseDouble(args[2]);
            } catch (NumberFormatException e) {
                System.err.println("Argument" + args[2] + " must be a double. Using 0.5 instead");
                gamma = 0.5;
            }
        }

        boolean isTest = Boolean.FALSE;
        if (args.length > 3) {
            String testOption = args[3];
            if (testOption.equalsIgnoreCase("-test")) {
                isTest = Boolean.TRUE;
            }
        }

        try {
            execute(lambda, mu, gamma, isTest, inputLocation);
        } catch (IOException | NotEnoughDataException ex) {
            LOGGER.warn(ex.getMessage());
        }
    }

    /**
     *
     * @param lambda
     * @param mu
     * @param gamma
     * @param isTest
     * @param inputLocation Local: "/Users/kjohnston/Google
     * Drive/VarStarData/UCR"
     * @throws IOException
     * @throws NotEnoughDataException
     */
    public static void execute(double lambda, double mu, double gamma, boolean isTest, String inputLocation)
            throws IOException, NotEnoughDataException {

        // Pull Data
        ReadingInUCRData starLight = new ReadingInUCRData(inputLocation);

        TimeDomainAttributeMaps trainingData = starLight.getTrainData("StarLightCurves");
        TimeDomainAttributeMaps testingData = starLight.getTestData("StarLightCurves");

        generatePatterns(trainingData, testingData);

        if (!isTest) {
            trainL3ML("Matrix_UCR_" + String.valueOf(lambda)
                    + "_" + String.valueOf(mu) + "_" + String.valueOf(gamma) + ".mat",
                    lambda, mu, gamma);
        } else {
            testL3ML(19, lambda, mu, gamma, "Matrix_UCR.mat");
        }

    }

    private static void generatePatterns(TimeDomainAttributeMaps trainingData,
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

        // =======================================================
        TrainCrossGenerator trainTest
                = new TrainCrossGenerator(trainingData.getSetOfClasses(), RAND);

        crossvalMap = trainTest.getCrossvalMap();

        // =======================================================
        Map<Integer, Map<String, RealMatrix>> setOfPatterns_Training_Tmp
                = new HashMap<>(trainingData.getSetOfClasses().keySet().size());

        setOfClasses_Training = trainingData.getSetOfClasses();

        for (Integer idx : trainingData.getSetOfClasses().keySet()) {

            Map<String, RealMatrix> pattern = new HashMap<>();

            Real2DCurve currentWaveform = trainingData.getSetOfWaveforms().get(idx);

            // statistics are on the amplitude of the waveform so phasing doesn't matter
            DescriptiveStatistics descript = new DescriptiveStatistics(currentWaveform
                    .getYArrayPrimitive());

            // input waveform is already phased for UCR data
            RealMatrix df = dfGenerator.evaluate(currentWaveform);

            pattern.put(ReadingInLINEARData.Views.DF.toString(), df);

            // input waveform doesn't require phasing
//            RealMatrix ssmm = ssmmGenerator.evaluate(currentWaveform);
//
//            pattern.put(ReadingInLINEARData.Views.SSMM.toString(), ssmm);
//
            RealVector statistics = new ArrayRealVector(
                    new double[]{descript.getMean(), descript.getStandardDeviation(),
                        descript.getSkewness(), descript.getKurtosis()});

            pattern.put(ReadingInLINEARData.Views.Statistics.toString(),
                    MatrixUtils.createColumnRealMatrix(statistics.toArray()));

            setOfPatterns_Training_Tmp.put(idx, pattern);
        }

        transformationVectors = NormalizeData.normalizeMultiViewMatrixVariate(setOfPatterns_Training_Tmp);

        setOfPatterns_Training = NormalizeData.applyNormalizeMatrixVariate(setOfPatterns_Training_Tmp,
                transformationVectors);

        // ======================================================
        //  Testing Data
        setOfClasses_Testing = testingData.getSetOfClasses();
        Map<Integer, Map<String, RealMatrix>> setOfPatterns_Testing_Tmp = new HashMap<>();

        for (Integer idx : testingData.getSetOfWaveforms().keySet()) {

            Map<String, RealMatrix> pattern = new HashMap<>();

            Real2DCurve currentWaveform = testingData.getSetOfWaveforms().get(idx);

            // statistics are on the amplitude of the waveform so phasing doesn't matter
            DescriptiveStatistics descript = new DescriptiveStatistics(currentWaveform
                    .getYArrayPrimitive());

            // input waveform is already phased for UCR data
            RealMatrix df = dfGenerator.evaluate(currentWaveform);

            pattern.put(ReadingInLINEARData.Views.DF.toString(), df);

            // input waveform doesn't require phasing
//            RealMatrix ssmm = ssmmGenerator.evaluate(currentWaveform);
//
//            pattern.put(ReadingInLINEARData.Views.SSMM.toString(), ssmm);


            RealVector statistics = new ArrayRealVector(
                    new double[]{descript.getMean(), descript.getStandardDeviation(),
                        descript.getSkewness(), descript.getKurtosis()});

            pattern.put(ReadingInLINEARData.Views.Statistics.toString(),
                    MatrixUtils.createColumnRealMatrix(statistics.toArray()));

            setOfPatterns_Testing_Tmp.put(idx, pattern);

        }

        setOfPatterns_Testing = NormalizeData.applyNormalizeMatrixVariate(setOfPatterns_Testing_Tmp,
                transformationVectors);

    }
    
}
