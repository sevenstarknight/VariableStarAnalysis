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

import fit.astro.vsa.analysis.ProcessTimeDomainViaL3ML;
import fit.astro.vsa.analysis.df.generators.DFGenerator;
import fit.astro.vsa.analysis.df.generators.DFOptions;
import fit.astro.vsa.analysis.ssmm.generators.SSMMGenerator;
import fit.astro.vsa.common.bindings.ml.TimeDomainAttributeMaps;
import fit.astro.vsa.common.bindings.math.Real2DCurve;
import fit.astro.vsa.utilities.ml.ecva.CanonicalVariates;
import fit.astro.vsa.utilities.ml.ecva.ECVA;
import fit.astro.vsa.common.utilities.io.ReadingInUCRData;
import fit.astro.vsa.common.utilities.io.dataset.ReadingInLINEARData;
import fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException;
import fit.astro.vsa.common.utilities.math.linearalgebra.MatrixOperations;
import fit.astro.vsa.utilities.ml.training.NormalizeData;
import fit.astro.vsa.common.datahandling.training.TrainCrossGenerator;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.apache.commons.math3.linear.ArrayRealVector;
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
public class ProcessUCRDataViaL3ML extends ProcessTimeDomainViaL3ML {

    private static final Random RAND = new Random(42L);

    private static final Logger LOGGER
            = LoggerFactory.getLogger(ProcessUCRDataViaL3ML.class);

    // =======================================================
    private static CanonicalVariates canonicalVariates_SSMM;
    private static CanonicalVariates canonicalVariates_DF;

    private static Pair<Map<String, RealVector>, Map<String, RealVector>> transformationVectors;

    /**
     * double[] tauArray = new double[]{0.25, 1.0, 1.75};
     * <p>
     * double[] muArray = new double[]{2.0, 5.0, 8.0};
     * <p>
     * double[] lambdaArray = new double[]{0.1, 0.5, 1.0};
     *
     * @param args
     */
    public static void main(String[] args) {
        String inputLocation = "UCR";

        double tau = 0.1;
        if (args.length > 0) {
            try {
                tau = Double.parseDouble(args[0]);
            } catch (NumberFormatException e) {
                System.err.println("Argument" + args[0] + " must be a double. Using 0.1 instead");
                tau = 0.1;
            }
        }

        double mu = 0.1;
        if (args.length > 1) {
            try {
                mu = Double.parseDouble(args[1]);
            } catch (NumberFormatException e) {
                System.err.println("Argument" + args[1] + " must be a double. Using 0.1 instead");
                mu = 0.1;
            }
        }

        double lambda = 0.1;
        if (args.length > 2) {
            try {
                lambda = Double.parseDouble(args[2]);
            } catch (NumberFormatException e) {
                System.err.println("Argument" + args[2] + " must be a double. Using 0.1 instead");
                lambda = 0.1;
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
            execute(tau, mu, lambda, isTest, inputLocation);
        } catch (IOException | NotEnoughDataException ex) {
            LOGGER.warn(ex.getMessage());
        }

    }

    /**
     *
     * @param tau
     * @param mu
     * @param lambda
     * @param isTest
     * @param location
     * @throws IOException
     * @throws NotEnoughDataException
     */
    public static void execute(double tau, double mu, double lambda, 
            boolean isTest, String location) throws IOException, NotEnoughDataException {

        // Pull Data
        ReadingInUCRData starLight = new ReadingInUCRData(location);

        TimeDomainAttributeMaps trainingData = starLight.getTrainData("StarLightCurves");
        TimeDomainAttributeMaps testingData = starLight.getTestData("StarLightCurves");

        generatePatterns(trainingData, testingData);

        // =================================================================
        if (!isTest) {
            trainL3ML("Vector_UCR_" +  String.valueOf(tau)
                    + "_" + String.valueOf(mu) + "_" + String.valueOf(lambda) + ".mat",
                    tau, mu, lambda);
        } else {
            testL3ML(19, tau, mu, lambda, "Vector_UCR.mat");
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
        Map<Integer, RealVector> setOfPatterns_SSMM = new HashMap<>();
        Map<Integer, RealVector> setOfPatterns_DF = new HashMap<>();
        Map<Integer, RealVector> setOfPatterns_Stats = new HashMap<>();

        Map<Integer, String> setOfClasses = new HashMap<>();

        TrainCrossGenerator trainTest
                = new TrainCrossGenerator(trainingData.getSetOfClasses(), RAND);

        crossvalMap = trainTest.getCrossvalMap();

        for (Integer idx : trainingData.getSetOfClasses().keySet()) {

            Real2DCurve currentWaveform = trainingData.getSetOfWaveforms().get(idx);

            // statistics are on the amplitude of the waveform so phasing doesn't matter
            DescriptiveStatistics descript = new DescriptiveStatistics(currentWaveform
                    .getYArrayPrimitive());

            // input waveform is already phased for UCR data
            RealMatrix df = dfGenerator.evaluate(currentWaveform);

            setOfPatterns_DF.put(idx, (new ArrayRealVector(
                    MatrixOperations.unpackMatrix(df))));

            // input waveform doesn't require phasing
            RealMatrix ssmm = ssmmGenerator.evaluate(currentWaveform);

            setOfPatterns_SSMM.put(idx, new ArrayRealVector(
                    MatrixOperations.unpackMatrix(ssmm)));
            RealVector statistics = new ArrayRealVector(
                    new double[]{descript.getMean(), descript.getStandardDeviation(),
                        descript.getSkewness(), descript.getKurtosis()});

            setOfPatterns_Stats.put(idx, statistics);

            setOfClasses.put(idx, trainingData.getSetOfClasses().get(idx));

        }

        // ======================================================
        // Training Data
        List<Integer> keysTraining = trainTest.getTrainingData();

        setOfPatterns_Training = new HashMap<>();

        setOfClasses_Training = keysTraining.stream().filter(setOfClasses::containsKey)
                .collect(Collectors.toMap(Function.identity(), setOfClasses::get));

        ECVA ecva_SSMM = new ECVA(keysTraining.stream()
                .filter(setOfPatterns_SSMM::containsKey)
                .collect(Collectors.toMap(Function.identity(), setOfPatterns_SSMM::get)),
                setOfClasses_Training);

        canonicalVariates_SSMM = ecva_SSMM.execute();
        ECVA ecva_DF = new ECVA(keysTraining.stream()
                .filter(setOfPatterns_DF::containsKey)
                .collect(Collectors.toMap(Function.identity(), setOfPatterns_DF::get)),
                setOfClasses_Training);

        canonicalVariates_DF = ecva_DF.execute();

        // ======================================================
        // Feature Space Multi-View
        for (Integer idx : trainTest.getTrainingData()) {

            setOfPatterns_Training.put(idx, new HashMap<>());
            setOfPatterns_Training.get(idx).put(ReadingInLINEARData.Views.SSMM.toString(),
                    setOfPatterns_SSMM.get(idx));
            setOfPatterns_Training.get(idx).put(ReadingInLINEARData.Views.DF.toString(),
                    setOfPatterns_DF.get(idx));
            setOfPatterns_Training.get(idx).put(ReadingInLINEARData.Views.Statistics.toString(),
                    setOfPatterns_Stats.get(idx));

        }

        transformationVectors = NormalizeData.normalizeMultiViewVectorVariate(setOfPatterns_Training);

        setOfPatterns_Training = NormalizeData.applyNormalizeVectorVariate(setOfPatterns_Training,
                transformationVectors);

        // ======================================================
        //  Testing Data
        setOfPatterns_Testing = new HashMap<>();
        setOfClasses_Testing = testingData.getSetOfClasses();

        for (Integer idx : testingData.getSetOfWaveforms().keySet()) {

            Real2DCurve currentWaveform = testingData.getSetOfWaveforms().get(idx);

            // statistics are on the amplitude of the waveform so phasing doesn't matter
            DescriptiveStatistics descript = new DescriptiveStatistics(currentWaveform
                    .getYArrayPrimitive());

            // input waveform is already phased for UCR data
            RealMatrix df = dfGenerator.evaluate(currentWaveform);
            RealVector df_Vector = canonicalVariates_DF.applyECVA(new ArrayRealVector(
                    MatrixOperations.unpackMatrix(df)));

            // input waveform doesn't require phasing
            RealMatrix ssmm = ssmmGenerator.evaluate(currentWaveform);

            RealVector ssmm_Vector = canonicalVariates_SSMM.applyECVA(new ArrayRealVector(
                    MatrixOperations.unpackMatrix(ssmm)));
            
            RealVector statistics = new ArrayRealVector(
                    new double[]{descript.getMean(), descript.getStandardDeviation(),
                        descript.getSkewness(), descript.getKurtosis()});

            setOfPatterns_Testing.put(idx, new HashMap<>());
            setOfPatterns_Testing.get(idx).put(ReadingInLINEARData.Views.SSMM.toString(), ssmm_Vector);
            setOfPatterns_Testing.get(idx).put(ReadingInLINEARData.Views.DF.toString(), df_Vector);
            setOfPatterns_Testing.get(idx).put(ReadingInLINEARData.Views.Statistics.toString(), statistics);

        }

        setOfPatterns_Testing = NormalizeData.applyNormalizeVectorVariate(setOfPatterns_Testing,
                transformationVectors);

    }

}
