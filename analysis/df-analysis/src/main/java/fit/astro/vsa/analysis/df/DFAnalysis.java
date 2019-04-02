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
package fit.astro.vsa.analysis.df;

import fit.astro.vsa.analysis.df.generators.DFGenerator;
import fit.astro.vsa.analysis.df.generators.DFOptions;
import fit.astro.vsa.common.bindings.math.Real2DCurve;
import fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException;
import fit.astro.vsa.common.utilities.math.linearalgebra.VectorOperations;
import fit.astro.vsa.utilities.ml.training.StandardTrainingImplementation;
import fit.astro.vsa.common.datahandling.training.TrainCrossGenerator;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class DFAnalysis {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(DFAnalysis.class);

    private Random RAND = new Random();

    private final Map<Integer, Real2DCurve> setOfPhasedData;
    private final Map<Integer, String> setOfClasses;

    private final List<Integer> resolutionArrayX = new ArrayList<>();
    private final List<Integer> resolutionArrayY = new ArrayList<>();

    private int intClusters = 6;

    /**
     *
     * @param setOfPhasedData
     * @param setOfClasses
     */
    public DFAnalysis(Map<Integer, Real2DCurve> setOfPhasedData,
            Map<Integer, String> setOfClasses) {
        this.setOfPhasedData = setOfPhasedData;
        this.setOfClasses = setOfClasses;

        resolutionArrayX.addAll(Arrays.asList(ArrayUtils.toObject(VectorOperations.linearSpace(40, 20, 5))));

        resolutionArrayY.addAll(Arrays.asList(ArrayUtils.toObject(VectorOperations.linearSpace(40, 20, 5))));
    }

     /**
     *
     * @return @throws
     * fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException
     */
    public List<RealVector> optimizeDF_Dependent() throws NotEnoughDataException {

        List<RealVector> errorArray = new ArrayList<>();

        TrainCrossGenerator trainCrossGenerator
                = new TrainCrossGenerator(
                        setOfClasses, RAND);

        StandardTrainingImplementation implementation = new StandardTrainingImplementation(
                        setOfClasses);
        
        // Loop Over Variables
        for (Integer resolutionX : resolutionArrayX) {

            for (Integer resolutionY : resolutionArrayY) {

                LOGGER.info("At Resolution X: " + resolutionX
                        + "  and Resolution Y: " + resolutionY);

                DFOptions dFOptions = new DFOptions(resolutionX, resolutionY,
                        new int[]{7, 1}, 0.2, DFOptions.Directions.both);

                // ============================================================
                // Generate the MC 
                DFGenerator dfGenerator
                        = new DFGenerator(dFOptions);

                RealVector errorRate = implementation.
                        execute_ClassDependent(setOfPhasedData, trainCrossGenerator, dfGenerator);

                
                errorArray.add((new ArrayRealVector(
                        new double[]{resolutionX, resolutionY})).append(errorRate));

            }

        }
        return errorArray;
    }
    
    /**
     *
     * @return @throws
     * fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException
     */
    public List<RealVector> optimizeDF() throws NotEnoughDataException {

        List<RealVector> errorArray = new ArrayList<>();

        TrainCrossGenerator trainCrossGenerator
                = new TrainCrossGenerator(
                        setOfClasses, RAND);

        StandardTrainingImplementation implementation = new StandardTrainingImplementation(
                        setOfClasses);
        
        // Loop Over Variables
        for (Integer resolutionX : resolutionArrayX) {

            for (Integer resolutionY : resolutionArrayY) {

                LOGGER.info("At Resolution X: " + resolutionX
                        + "  and Resolution Y: " + resolutionY);

                DFOptions dFOptions = new DFOptions(resolutionX, resolutionY,
                        new int[]{7, 1}, 0.2, DFOptions.Directions.both);

                // ============================================================
                // Process all training data
                Map<Integer, RealMatrix> featureSpace = new HashMap<>();

                // Generate the MC 
                DFGenerator dfGenerator
                        = new DFGenerator(dFOptions);

                for (Integer e : setOfPhasedData.keySet()) {

                    Real2DCurve currentWaveform = setOfPhasedData.get(e);

                    // Generate the MC 
                    RealMatrix df = dfGenerator
                            .evaluate(currentWaveform);

                    featureSpace.put(e, df);
                }

                RealVector errorRate = implementation.
                        execute_ClassIndependent(featureSpace, trainCrossGenerator);

                
                errorArray.add((new ArrayRealVector(
                        new double[]{resolutionX, resolutionY})).append(errorRate));

            }

        }
        return errorArray;
    }

    /**
     * Fix the RAND to be used
     *
     * @param RAND
     */
    public void setRAND(Random RAND) {
        this.RAND = RAND;
    }

    /**
     *
     * @param intClusters
     */
    public void setIntClusters(int intClusters) {
        this.intClusters = intClusters;
    }

}
