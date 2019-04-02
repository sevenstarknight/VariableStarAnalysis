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

import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;
import fit.astro.vsa.analysis.ssmm.generators.SSMMGenerator;
import fit.astro.vsa.common.bindings.analysis.astro.VarStarDataset;
import fit.astro.vsa.common.bindings.math.Real2DCurve;
import fit.astro.vsa.utilities.ml.ecva.CanonicalVariates;
import fit.astro.vsa.utilities.ml.ecva.ECVA;
import fit.astro.vsa.common.utilities.io.MatlabFunctions;
import fit.astro.vsa.common.utilities.io.dataset.ReadingInLINEARData;
import fit.astro.vsa.common.utilities.math.linearalgebra.MatrixOperations;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

/**
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class GenerateSSMM_FeatureSpace_LINEAR {

    public static void main(String[] args) throws IOException {

        String fileLocation = "/Users/kjohnston/Google Drive/VarStarData/LinearData/LinearData.mat";

        ReadingInLINEARData lINEARData = new ReadingInLINEARData(fileLocation);

        VarStarDataset dataset = lINEARData.getLinearDataset();

        Map<Integer, String> setOfClasses = dataset.getMultiViewSideData().getSetOfClasses();
        // =======================================================
        double windowWidth = SSMMGenerator.estimateWindowWidth(
                dataset.getTimeD().values().iterator().next(), 35);

        SSMMGenerator ssmmGenerator
                = new SSMMGenerator(0.06, windowWidth);

        Map<Integer, RealVector> setOfPatterns = new HashMap<>();

        for (Integer idx : dataset.getTimeD().keySet()) {

            Real2DCurve currentWaveform = dataset.getTimeD().get(idx);

            RealMatrix ssmm = ssmmGenerator.evaluate(currentWaveform);

            DescriptiveStatistics descript = new DescriptiveStatistics(currentWaveform
                    .getYArrayPrimitive());

            setOfPatterns.put(idx, (new ArrayRealVector(
                    MatrixOperations.unpackMatrix(ssmm)))
                    .append(descript.getMean())
                    .append(descript.getStandardDeviation()));

        }

        ECVA ecva = new ECVA(setOfPatterns, setOfClasses);

        CanonicalVariates canonicalVariates = ecva.execute();

        // =================================================================
        MLCell labelsML = new MLCell("SSMM_Labels",
                new int[]{canonicalVariates.getCanonicalVariates().size(), 1});
        
        RealMatrix cvaMatrix = new Array2DRowRealMatrix(
                canonicalVariates.getCanonicalVariates().size(), 4);
        
        int counter = 0;
        for (Integer idx : canonicalVariates.getCanonicalVariates().keySet()) {

            RealVector cva = canonicalVariates.getCanonicalVariates().get(idx);
            cvaMatrix.setRowVector(counter, cva);
            
            MLChar label = new MLChar("label", setOfClasses.get(idx));
            labelsML.set(label, counter);
            
            counter++;
        }
        
        MLArray errorML = new MLDouble("SSMM_CVA", cvaMatrix.getData());
        List<MLArray> list = new ArrayList<>();
        list.add(errorML);
        list.add(labelsML);

        MatlabFunctions.storeToTestAnalysis("SSMM_CVAData_LINEAR.mat", list);
    }
}
