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

import fit.astro.vsa.analysis.ProcessRawMultiViewKNN_MV;
import fit.astro.vsa.analysis.df.generators.DFGenerator;
import fit.astro.vsa.analysis.df.generators.DFOptions;
import fit.astro.vsa.analysis.ssmm.generators.SSMMGenerator;
import fit.astro.vsa.common.bindings.analysis.astro.VarStarDataset;
import fit.astro.vsa.common.bindings.math.Real2DCurve;
import fit.astro.vsa.common.datahandling.training.TrainCrossTestGenerator;
import fit.astro.vsa.common.utilities.io.dataset.ReadingInLINEARData;
import fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException;
import fit.astro.vsa.analysis.raw.SupportFunctionality;
import fit.astro.vsa.utilities.ml.training.NormalizeData;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author kjohnston
 */
public class RawKnn_LINEAR_Matrix extends ProcessRawMultiViewKNN_MV {

    private static final Random RAND = new Random(42L);

    private static final Logger LOGGER
            = LoggerFactory.getLogger(RawKnn_LINEAR_Matrix.class);

    private static VarStarDataset dataset;

    private static Pair<Map<String, RealMatrix>, Map<String, RealMatrix>> transformationVectors;

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

        Map<Integer, Map<String, RealMatrix>> setOfPatterns = new HashMap<>(setOfClasses.keySet().size());

        for (Integer idx : setOfClasses.keySet()) {

            Map<String, RealMatrix> pattern = new HashMap<>();

            Real2DCurve phasedWaveform = phasedData.get(idx);
            Real2DCurve currentWaveform = dataset.getTimeD().get(idx);

            // ===========
            RealMatrix df = dfGenerator.evaluate(phasedWaveform);

            pattern.put(ReadingInLINEARData.Views.DF.toString(), df);

            // ===========
            RealMatrix ssmm = ssmmGenerator.evaluate(currentWaveform);

            pattern.put(ReadingInLINEARData.Views.SSMM.toString(), ssmm);

            pattern.put(ReadingInLINEARData.Views.Photometric.toString(), 
                    MatrixUtils.createColumnRealMatrix(dataset.getMultiViewSideData()
                    .getMapOfVectorPatterns().get(idx).get(ReadingInLINEARData.Views.Photometric.toString()).toArray()));

            pattern.put(ReadingInLINEARData.Views.TimeDomainVar.toString(), 
                    MatrixUtils.createColumnRealMatrix(dataset.getMultiViewSideData()
                    .getMapOfVectorPatterns().get(idx).get(ReadingInLINEARData.Views.TimeDomainVar.toString()).toArray()));
            

            setOfPatterns.put(idx, pattern);
            setOfClasses.put(idx, setOfClasses.get(idx));

        }

        TrainCrossTestGenerator trainTest
                = new TrainCrossTestGenerator(setOfClasses, 0.5, RAND);

        crossvalMap = trainTest.getCrossvalMap();

        //=============================================================
        List<Integer> keysTraining = trainTest.getTrainingData();

        Map<Integer, Map<String, RealMatrix>> setOfPatterns_Training_Tmp
                = keysTraining.stream().filter(setOfPatterns::containsKey)
                        .collect(Collectors.toMap(Function.identity(), setOfPatterns::get));

        transformationVectors = NormalizeData.normalizeMultiViewMatrixVariate(setOfPatterns_Training_Tmp);

        setOfPatterns_Training = NormalizeData.applyNormalizeMatrixVariate(setOfPatterns_Training_Tmp,
                transformationVectors);

        setOfClasses_Training = keysTraining.stream().filter(setOfClasses::containsKey)
                .collect(Collectors.toMap(Function.identity(), setOfClasses::get));

        //=============================================================
        List<Integer> keysTesting = trainTest.getTestingData();

        Map<Integer, Map<String, RealMatrix>> setOfPatterns_Testing_Tmp
                = keysTesting.stream().filter(setOfPatterns::containsKey)
                        .collect(Collectors.toMap(Function.identity(), setOfPatterns::get));

        setOfPatterns_Testing = NormalizeData.applyNormalizeMatrixVariate(setOfPatterns_Testing_Tmp,
                transformationVectors);

        setOfClasses_Testing = keysTesting.stream().filter(setOfClasses::containsKey)
                .collect(Collectors.toMap(Function.identity(), setOfClasses::get));

    }


}
