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
package fit.astro.vsa.analysis.ssmm;

import fit.astro.vsa.analysis.ssmm.generators.SSMMGenerator;
import fit.astro.vsa.common.bindings.math.Real2DCurve;
import fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException;
import fit.astro.vsa.common.utilities.math.linearalgebra.VectorOperations;
import fit.astro.vsa.utilities.ml.training.StandardTrainingImplementation;
import fit.astro.vsa.common.datahandling.training.TrainCrossGenerator;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class SSMMAnalysis {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(SSMMAnalysis.class);

    private Random RAND = new Random();

    private final Map<Integer, Real2DCurve> setOfWaveforms;
    private final Map<Integer, String> setOfClasses;

    private final List<Double> resolutionArray = new ArrayList<>();
    private final List<Double> scaleArray = new ArrayList<>();

    /**
     *
     * @param setOfWaveforms
     * @param setOfClasses
     */
    public SSMMAnalysis(
            Map<Integer, Real2DCurve> setOfWaveforms,
            Map<Integer, String> setOfClasses) {
        this.setOfWaveforms = setOfWaveforms;
        this.setOfClasses = setOfClasses;

        resolutionArray.addAll(Arrays.asList(ArrayUtils.toObject(VectorOperations.linearSpace(0.22, 0.02, 0.02))));

        scaleArray.addAll(Arrays.asList(ArrayUtils.toObject(VectorOperations.linearSpace(60.0, 20.0, 2.0))));
    }

    /**
     *
     * @return
     * @throws fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException
     */
    public List<RealVector> optimizeSSMM_Dependent() throws NotEnoughDataException {


        List<RealVector> errorArray = new ArrayList<>();

        TrainCrossGenerator trainCrossGenerator
                = new TrainCrossGenerator(setOfClasses, RAND);

        StandardTrainingImplementation implementation = new StandardTrainingImplementation(
                        setOfClasses);
        
        // Loop Over Variables
        for (Double scale : scaleArray) {
            for (Double resolution : resolutionArray) {

                LOGGER.info("At Scale: " + scale
                        + "  and Resolution: " + resolution);

                Double[] doubleArray = ArrayUtils.toObject(VectorOperations.linearSpace(2, -2, resolution));

                List<Double> states = Arrays.asList(doubleArray);

                // ============================================================
                // Generate the MC 
                double windowWidth = SSMMGenerator.estimateWindowWidth(
                        setOfWaveforms.values().iterator().next(), scale);

                RealVector errorRate = implementation
                        .execute_ClassDependent(setOfWaveforms, trainCrossGenerator, 
                                new SSMMGenerator(states, windowWidth));

                errorArray.add((new ArrayRealVector(
                        new double[]{resolution, scale}).append(errorRate)));

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
     * @return
     */
    public List<Double> getResolutionArray() {
        return resolutionArray;
    }

}
