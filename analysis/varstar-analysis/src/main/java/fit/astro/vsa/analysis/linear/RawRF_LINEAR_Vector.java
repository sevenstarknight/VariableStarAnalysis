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
package fit.astro.vsa.analysis.linear;

import fit.astro.vsa.analysis.ProcessRawRF;
import fit.astro.vsa.analysis.df.generators.DFGenerator;
import fit.astro.vsa.analysis.df.generators.DFOptions;
import fit.astro.vsa.analysis.ssmm.generators.SSMMGenerator;
import fit.astro.vsa.common.bindings.analysis.astro.VarStarDataset;
import fit.astro.vsa.common.bindings.math.Real2DCurve;
import fit.astro.vsa.common.datahandling.training.TrainCrossTestGenerator;
import fit.astro.vsa.common.utilities.io.dataset.ReadingInLINEARData;
import fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException;
import fit.astro.vsa.common.utilities.math.linearalgebra.MatrixOperations;
import fit.astro.vsa.analysis.raw.SupportFunctionality;
import fit.astro.vsa.utilities.ml.training.NormalizeData;
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
 * @author kjohnston
 */
public class RawRF_LINEAR_Vector extends ProcessRawRF {

    private static final Random RAND = new Random(42L);

    private static final Logger LOGGER
            = LoggerFactory.getLogger(RawRF_LINEAR_Vector.class);

    private static Pair<Map<String, RealVector>, Map<String, RealVector>> transformationVectors;

    private static VarStarDataset dataset;

    public static void main(String[] args) throws IOException, NotEnoughDataException {

        String fileLocation = "LinearData/LinearData.mat";

        // =================================================================
        double alpha = 001;
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

        execute(alpha, isTest, fileLocation);

    }

    public static void execute(double alpha, boolean isTest, String fileLocation) throws IOException, NotEnoughDataException {

        ReadingInLINEARData lINEARData = new ReadingInLINEARData(fileLocation);

        dataset = lINEARData.getLinearDataset();

        
//        // =================================================================
//        generatePatterns(ReadingInLINEARData.Views.Statistics.toString());
//
//        if (!isTest) {
//            trainRF("LINEAR" + ReadingInLINEARData.Views.Statistics.toString());
//        } else {
//            testRF(alpha, "LINEAR" + ReadingInLINEARData.Views.Statistics.toString());
//
//        }
//        
//        // =================================================================
//        generatePatterns(ReadingInLINEARData.Views.DF.toString());
//
//        if (!isTest) {
//            trainRF("LINEAR" + ReadingInLINEARData.Views.DF.toString());
//        } else {
//            testRF(alpha, "LINEAR" + ReadingInLINEARData.Views.DF.toString());
//
//        }

        // =================================================================
        generatePatterns(ReadingInLINEARData.Views.SSMM.toString());

        if (!isTest) {
            trainRF("LINEAR" + ReadingInLINEARData.Views.SSMM.toString());
        } else {
            testRF(alpha, "LINEAR" + ReadingInLINEARData.Views.SSMM.toString());

        }
    }

    /**
     *
     * @param phasedData
     * @param setOfClasses
     */
    private static void generatePatterns(String view) {

        Map<Integer, String> setOfClasses = dataset.getMultiViewSideData().getSetOfClasses();

        Map<Integer, Real2DCurve> phasedData = SupportFunctionality.generatePhasedData(dataset);

        // =======================================================
        // Feature Processing
        DFOptions dFOptions = new DFOptions(30, 25,
                new int[]{7, 1}, 0.4, DFOptions.Directions.both);

        DFGenerator dfGenerator
                = new DFGenerator(dFOptions);

        // =======================================================
        double windowWidth = SSMMGenerator.estimateWindowWidth(
                phasedData.values().iterator().next(), 35);

        SSMMGenerator ssmmGenerator
                = new SSMMGenerator(0.06, windowWidth);

        Map<Integer, Map<String, RealVector>> setOfPatterns = new HashMap<>(setOfClasses.keySet().size());

        for (Integer idx : setOfClasses.keySet()) {

            Map<String, RealVector> pattern = new HashMap<>();

            Real2DCurve phasedWaveform = phasedData.get(idx);
            Real2DCurve currentWaveform = dataset.getTimeD().get(idx);

            double period = dataset.getMultiViewSideData()
                    .getMapOfVectorPatterns().get(idx).get(
                    ReadingInLINEARData.Views.TimeDomainVar.toString()).getEntry(0);

            DescriptiveStatistics descript = new DescriptiveStatistics(currentWaveform
                    .getYArrayPrimitive());

            RealVector statistics = new ArrayRealVector(
                    new double[]{period, descript.getMean(), descript.getStandardDeviation(),
                        descript.getSkewness(), descript.getKurtosis()});

            pattern.put(ReadingInLINEARData.Views.Statistics.toString(), statistics);

            // input waveform is already phased for UCR data
            RealMatrix df = dfGenerator.evaluate(phasedWaveform);

            pattern.put(ReadingInLINEARData.Views.DF.toString(), new ArrayRealVector(
                    MatrixOperations.unpackMatrix(df)));

            // input waveform doesn't require phasing
            RealMatrix ssmm = ssmmGenerator.evaluate(currentWaveform);

            pattern.put(ReadingInLINEARData.Views.SSMM.toString(), new ArrayRealVector(
                    MatrixOperations.unpackMatrix(ssmm)));

            setOfPatterns.put(idx, pattern);
            setOfClasses.put(idx, setOfClasses.get(idx));

        }

        TrainCrossTestGenerator trainTest
                = new TrainCrossTestGenerator(setOfClasses, 0.5, RAND);

        crossvalMap = trainTest.getCrossvalMap();

        //=============================================================
        List<Integer> keysTraining = trainTest.getTrainingData();

        Map<Integer, Map<String, RealVector>> setOfPatterns_Training_Tmp
                = keysTraining.stream().filter(setOfPatterns::containsKey)
                        .collect(Collectors.toMap(Function.identity(), setOfPatterns::get));

        transformationVectors = NormalizeData.normalizeMultiViewVectorVariate(setOfPatterns_Training_Tmp);

        setOfClasses_Training = keysTraining.stream().filter(setOfClasses::containsKey)
                .collect(Collectors.toMap(Function.identity(), setOfClasses::get));

        Map<Integer, Map<String, RealVector>> setOfPatterns_TrainingTmp = 
                NormalizeData.applyNormalizeVectorVariate(setOfPatterns_Training_Tmp,
                transformationVectors);
        
        
        setOfPatterns_Training = new HashMap<>();
        setOfPatterns_TrainingTmp.keySet().forEach((idx) -> {
            setOfPatterns_Training.put(idx, 
                    setOfPatterns_TrainingTmp.get(idx).get(view));
        });
        
        //=============================================================
        List<Integer> keysTesting = trainTest.getTestingData();

        Map<Integer, Map<String, RealVector>> setOfPatterns_Testing_Tmp
                = keysTesting.stream().filter(setOfPatterns::containsKey)
                        .collect(Collectors.toMap(Function.identity(), setOfPatterns::get));

        setOfClasses_Testing = keysTesting.stream().filter(setOfClasses::containsKey)
                .collect(Collectors.toMap(Function.identity(), setOfClasses::get));

        Map<Integer, Map<String, RealVector>> setOfPatterns_TestingTmp = 
                NormalizeData.applyNormalizeVectorVariate(setOfPatterns_Testing_Tmp,
                transformationVectors);
        
        setOfPatterns_Testing= new HashMap<>();
        setOfPatterns_TestingTmp.keySet().forEach((idx) -> {
            setOfPatterns_Testing.put(idx, 
                    setOfPatterns_TestingTmp.get(idx).get(view));
        });
        
    }


}
