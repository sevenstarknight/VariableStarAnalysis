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

package fit.astro.vsa.analysis.feature;

import java.util.Arrays;
import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;

/**
 * Wrapper for the apache math functions specifically for FFT usage
 * 
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class ApacheFFT {

    //Empty Constructor
    private ApacheFFT(){
        
    }
    
    /**
     * Provides rapper for org.apache.commons.math3.transform.FastFourierTransformer
     * Generates the PSD
     * @param input Waveform
     * @return 
     */
    public static double[] generatePSD(double input[]) {

        //fft works on data length = some power of two
        int fftLength;
        int length = input.length;  //initialized with input's length
        int power = 0;
        while (true) {
            int powOfTwo = (int) Math.pow(2, power);
            //maximum no. of values to be applied fft on

            if (powOfTwo == length) {
                fftLength = powOfTwo;
                break;
            }
            if (powOfTwo > length) {
                fftLength = (int) Math.pow(2, (power - 1));
                break;
            }
            power++;
        }

        double[] tempInput = Arrays.copyOf(input, fftLength);
        FastFourierTransformer fft = new FastFourierTransformer(
                DftNormalization.STANDARD);
        //apply fft on input
        Complex[] complexTransInput = fft.transform(tempInput,
                TransformType.FORWARD);

        for (int i = 0; i < complexTransInput.length; i++) {
            double real = (complexTransInput[i].getReal());
            double img = (complexTransInput[i].getImaginary());

            tempInput[i] = Math.sqrt((real * real) + (img * img));
        }

        return tempInput;
    }
}
