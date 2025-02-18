import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-loan-application-form',
  standalone: true,
  imports: [FormsModule, CommonModule],
  templateUrl: './loan-application-form.component.html',
  styleUrls: ['./loan-application-form.component.css'],
})
export class LoanApplicationFormComponent {
  dummyPayload: any = null;
  formData = {
    OCCUPATION_TYPE: '',
    AMT_INCOME_TOTAL: 0, // annual income
    AMT_CREDIT: 0, // requested credit
    loanTermMonths: 0, // map from pos_cash_balance_data[0].CNT_INSTALMENT
    EXT_SOURCE_2: 0, // external source score
  };

  predictionResult: any = null;

  // Pre-populate a dropdown for Occupation, if desired
  occupations: string[] = [
    'Laborers',
    'Core staff',
    'Accountants',
    'Managers',
    'Drivers',
    'Sales staff',
    'Cleaning staff',
    'Cooking staff',
    'Private service staff',
    'Medicine staff',
    'Security staff',
    'High skill tech staff',
    'Waiters/barmen staff',
    'Low-skill Laborers',
    'Realty agents',
    'Secretaries',
    'IT staff',
    'HR staff',
  ];

  constructor(private http: HttpClient) {}

  generateDummyData() {
    this.http
      .get<any>('http://localhost:5001/generate_dummy')
      .subscribe((response) => {
        // Store entire nested object so we keep all 142 features
        this.dummyPayload = response;

        // Also map just the 5 fields we want into the form
        this.formData = {
          OCCUPATION_TYPE: response.application.OCCUPATION_TYPE ?? '',
          AMT_INCOME_TOTAL: response.application.AMT_INCOME_TOTAL ?? 0,
          AMT_CREDIT: response.application.AMT_CREDIT ?? 0,
          loanTermMonths:
            response.pos_cash_balance_data?.[0]?.CNT_INSTALMENT ?? 0,
          EXT_SOURCE_2: response.application.EXT_SOURCE_2 ?? 0,
        };
      });
  }

  predictLoan() {
    if (!this.dummyPayload) {
      console.error('No dummy payload available. Generate first.');
      return;
    }

    const RUBLE_TO_USD = 90;
    const rubIncome = Math.round(this.formData.AMT_INCOME_TOTAL * RUBLE_TO_USD);
    const rubCredit = Math.round(this.formData.AMT_CREDIT * RUBLE_TO_USD);

    this.dummyPayload.application.OCCUPATION_TYPE =
      this.formData.OCCUPATION_TYPE;
    this.dummyPayload.application.AMT_INCOME_TOTAL = rubIncome;
    this.dummyPayload.application.AMT_CREDIT = rubCredit;

    if (this.dummyPayload.pos_cash_balance_data?.[0]) {
      this.dummyPayload.pos_cash_balance_data[0].CNT_INSTALMENT =
        this.formData.loanTermMonths;
    }
    this.dummyPayload.application.EXT_SOURCE_2 = this.formData.EXT_SOURCE_2;

    // 🟡 Log the final object to see the actual ruble amounts:
    console.log('[DEBUG] Final payload to /predict:', this.dummyPayload);

    this.http
      .post<any>('http://localhost:5001/predict', this.dummyPayload)
      .subscribe(
        (response) => {
          this.predictionResult = response;
        },
        (error) => {
          console.error('Error predicting loan:', error);
        }
      );
  }

  getProbabilityNumber(): number {
    return parseFloat(this.predictionResult.formatted_probability);
  }
}
