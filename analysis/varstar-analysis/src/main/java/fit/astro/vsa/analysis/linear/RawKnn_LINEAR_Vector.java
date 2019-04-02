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

import fit.astro.vsa.analysis.ProcessRawMultiViewKNN;
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
import fit.astro.vsa.utilities.ml.ecva.ECVA;
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
public class RawKnn_LINEAR_Vector extends ProcessRawMultiViewKNN {

    private static final Random RAND = new Random(42L);

    private static final Logger LOGGER
            = LoggerFactory.getLogger(RawKnn_LINEAR_Vector.class);

    private static Pair<Map<String, RealVector>, Map<String, RealVector>> transformationVectors;

    private static VarStarDataset dataset;

    public static void main(String[] args) throws IOException, NotEnoughDataException {

        String fileLocation = "LinearData/LinearData.mat";

        // =================================================================
        int kValue = 3;
        if (args.length > 0) {
            try {
                kValue = Integer.parseInt(args[0]);
            } catch (NumberFormatException e) {
                System.err.println("Argument" + args[0] + " must be a int. Using 3 instead");
                kValue = 3;
            }
        }

        boolean isTest = Boolean.FALSE;
        if (args.length > 1) {
            String testOption = args[1];
            if (testOption.equalsIgnoreCase("-test")) {
                isTest = Boolean.TRUE;
            }
        }

        execute(kValue, isTest, fileLocation);

    }

    public static void execute(int kValue, boolean isTest, String fileLocation) 
            throws IOException, NotEnoughDataException {

        ReadingInLINEARData lINEARData = new ReadingInLINEARData(fileLocation);

        dataset = lINEARData.getLinearDataset();

        generatePatterns();

        if (!isTest) {
            trainKNN("LINEAR");
        } else {
            testKNN(kValue, "LINEAR");

        }

        // =================================================================
    }

    /**
     *
     * @param phasedData
     * @param setOfClasses
     */
    private static void generatePatterns() {

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

        Map<Integer, RealVector> setOfPatterns_SSMM = new HashMap<>();
        Map<Integer, RealVector> setOfPatterns_DF = new HashMap<>();

        TrainCrossTestGenerator trainTest
                = new TrainCrossTestGenerator(setOfClasses, 0.5, RAND);

        crossvalMap = trainTest.getCrossvalMap();

        for (Integer idx : setOfClasses.keySet()) {

            Real2DCurve phasedWaveform = phasedData.get(idx);
            Real2DCurve currentWaveform = dataset.getTimeD().get(idx);
            
            // input waveform is already phased for UCR data
            RealMatrix df = dfGenerator.evaluate(phasedWaveform);

            setOfPatterns_DF.put(idx, new ArrayRealVector(
                    MatrixOperations.unpackMatrix(df)));

            // input waveform doesn't require phasing
            RealMatrix ssmm = ssmmGenerator.evaluate(currentWaveform);

            setOfPatterns_SSMM.put(idx, new ArrayRealVector(
                    MatrixOperations.unpackMatrix(ssmm)));

            setOfClasses.put(idx, setOfClasses.get(idx));

        }

        // ======================================================
        // Generate Training Data
        List<Integer> keysTraining = trainTest.getTrainingData();
        
        setOfClasses_Training = keysTraining.stream().filter(setOfClasses::containsKey)
                .collect(Collectors.toMap(Function.identity(), setOfClasses::get));

        // ======================================================
        // Featrure Space Multi-View
        Map<Integer, Map<String, RealVector>> setOfPatterns_Training_tmp = new HashMap<>();

        for (Integer idx : keysTraining) {

            setOfPatterns_Training_tmp.put(idx, new HashMap<>());
            setOfPatterns_Training_tmp.get(idx).put(ReadingInLINEARData.Views.SSMM.toString(), setOfPatterns_SSMM.get(idx));
            setOfPatterns_Training_tmp.get(idx).put(ReadingInLINEARData.Views.DF.toString(), setOfPatterns_DF.get(idx));

            setOfPatterns_Training_tmp.get(idx).put(ReadingInLINEARData.Views.Photometric.toString(), dataset.getMultiViewSideData()
                    .getMapOfVectorPatterns().get(idx).get(ReadingInLINEARData.Views.Photometric.toString()));

            setOfPatterns_Training_tmp.get(idx).put(ReadingInLINEARData.Views.TimeDomainVar.toString(), dataset.getMultiViewSideData()
                    .getMapOfVectorPatterns().get(idx).get(ReadingInLINEARData.Views.TimeDomainVar.toString()));

        }

        transformationVectors = NormalizeData.normalizeMultiViewVectorVariate(setOfPatterns_Training_tmp);

        setOfPatterns_Training = NormalizeData.applyNormalizeVectorVariate(setOfPatterns_Training_tmp,
                transformationVectors);

        // ======================================================
        List<Integer> keysTesting = trainTest.getTestingData();

        Map<Integer, Map<String, RealVector>> setOfPatterns_Testing_tmp = new HashMap<>();

        setOfClasses_Testing = keysTesting.stream().filter(setOfClasses::containsKey)
                .collect(Collectors.toMap(Function.identity(), setOfClasses::get));

        for (Integer idx : keysTesting) {

            setOfPatterns_Testing_tmp.put(idx, new HashMap<>());
            setOfPatterns_Testing_tmp.get(idx).put(ReadingInLINEARData.Views.SSMM.toString(), setOfPatterns_SSMM.get(idx));
            setOfPatterns_Testing_tmp.get(idx).put(ReadingInLINEARData.Views.DF.toString(), setOfPatterns_DF.get(idx));

            setOfPatterns_Testing_tmp.get(idx).put(ReadingInLINEARData.Views.Photometric.toString(), dataset.getMultiViewSideData()
                    .getMapOfVectorPatterns().get(idx).get(ReadingInLINEARData.Views.Photometric.toString()));

            setOfPatterns_Testing_tmp.get(idx).put(ReadingInLINEARData.Views.TimeDomainVar.toString(), dataset.getMultiViewSideData()
                    .getMapOfVectorPatterns().get(idx).get(ReadingInLINEARData.Views.TimeDomainVar.toString()));

        }

        setOfPatterns_Testing = NormalizeData.applyNormalizeVectorVariate(setOfPatterns_Testing_tmp,
                transformationVectors);

    }

}
