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

import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import fit.astro.vsa.common.bindings.ml.TimeDomainAttributeMaps;
import fit.astro.vsa.common.utilities.io.MatlabFunctions;
import fit.astro.vsa.common.utilities.io.ReadingInUCRData;
import fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class OptimizeDF_UCR {

    public static void main(String[] args) throws IOException, NotEnoughDataException {

        ReadingInUCRData starLight = new ReadingInUCRData(
                "/Users/kjohnston/Google Drive/VarStarData/UCR");

        TimeDomainAttributeMaps trainingData = starLight.getTrainData("StarLightCurves");
        TimeDomainAttributeMaps testingData = starLight.getTestData("StarLightCurves");
                
        // ===============================================================
        DFAnalysis analysis = new DFAnalysis(
                trainingData.getSetOfWaveforms(),
                trainingData.getSetOfClasses());
        
        List<RealVector> errorArray = analysis.optimizeDF_Dependent();

        RealMatrix errorMetric = new Array2DRowRealMatrix(
                errorArray.get(0).getDimension(),
                errorArray.size());

        for(int idx = 0; idx < errorArray.size(); idx++){
            errorMetric.setColumnVector(idx, errorArray.get(idx));
        }
        
        MLArray errorML = new MLDouble("error", errorMetric.getData());
        List<MLArray> list = new ArrayList<>();
        list.add(errorML);
        
        MatlabFunctions.storeToTestAnalysis("DF-Error-UCR_Dependent.mat", list);
    }
}
