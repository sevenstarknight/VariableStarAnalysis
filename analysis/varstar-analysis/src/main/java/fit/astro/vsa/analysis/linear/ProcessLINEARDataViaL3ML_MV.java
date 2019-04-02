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
package fit.astro.vsa.analysis.linear;

import fit.astro.vsa.analysis.ProcessTimeDomainViaL3ML_MV;
import fit.astro.vsa.analysis.df.generators.DFGenerator;
import fit.astro.vsa.analysis.df.generators.DFOptions;
import fit.astro.vsa.analysis.ssmm.generators.SSMMGenerator;
import fit.astro.vsa.common.bindings.analysis.astro.VarStarDataset;
import fit.astro.vsa.common.bindings.math.Real2DCurve;
import fit.astro.vsa.common.utilities.io.dataset.ReadingInLINEARData;
import fit.astro.vsa.common.utilities.io.dataset.ReadingInLINEARData.Views;
import fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException;
import fit.astro.vsa.analysis.raw.SupportFunctionality;
import fit.astro.vsa.utilities.ml.training.NormalizeData;
import fit.astro.vsa.common.datahandling.training.TrainCrossTestGenerator;
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
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class ProcessLINEARDataViaL3ML_MV extends ProcessTimeDomainViaL3ML_MV {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(ProcessLINEARDataViaL3ML_MV.class);

    private static final Random RAND = new Random(42L);

    private static VarStarDataset dataset;

    private static Pair<Map<String, RealMatrix>, Map<String, RealMatrix>> transformationVectors;

    // ==================================
    public static void main(String[] args) {
        String fileLocation = "LinearData/LinearData.mat";

        // =================================================================
        double lambda = 0.5;
        if (args.length > 0) {
            try {
                lambda = Double.parseDouble(args[0]);
            } catch (NumberFormatException e) {
                System.err.println("Argument" + args[0] + " must be a double. Using 0.25 instead");
                lambda = 0.5;
            }
        }

        double mu = 0.5;
        if (args.length > 1) {
            try {
                mu = Double.parseDouble(args[1]);
            } catch (NumberFormatException e) {
                System.err.println("Argument" + args[1] + " must be a double. Using 5.0 instead");
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
            execute(lambda, mu, gamma, isTest, fileLocation);
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
     * @param fileLocation
     * @throws IOException
     * @throws NotEnoughDataException
     */
    public static void execute(double lambda, double mu, double gamma, boolean isTest, String fileLocation) throws IOException,
            NotEnoughDataException {

        ReadingInLINEARData lINEARData = new ReadingInLINEARData(fileLocation);

        dataset = lINEARData.getLinearDataset();

        generatePatterns();

        // =================================================================
        if (!isTest) {
            trainL3ML("Matrix_LINEAR_" + String.valueOf(lambda)
                    + "_" + String.valueOf(mu) + "_" + String.valueOf(gamma) + ".mat",
                    lambda, mu, gamma);
        } else {
            testL3ML(15, lambda, mu, gamma, "Matrix_LINEAR.mat");
        }
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

            pattern.put(Views.Photometric.toString(),
                    MatrixUtils.createColumnRealMatrix(dataset.getMultiViewSideData()
                            .getMapOfVectorPatterns().get(idx).get(Views.Photometric.toString()).toArray()));

            pattern.put(Views.TimeDomainVar.toString(),
                    MatrixUtils.createColumnRealMatrix(dataset.getMultiViewSideData()
                            .getMapOfVectorPatterns().get(idx).get(Views.TimeDomainVar.toString()).toArray()));

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
