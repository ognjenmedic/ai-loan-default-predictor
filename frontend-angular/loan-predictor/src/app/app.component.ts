import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet } from '@angular/router';
import { LoanApplicationFormComponent } from './loan-application-form/loan-application-form.component';
import { NavbarComponent } from './navbar/navbar.component';
import { FooterComponent } from './footer/footer.component';
import { ButtonsComponent } from './buttons/buttons.component';
import { FaqComponent } from './faq/faq.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    RouterOutlet,
    LoanApplicationFormComponent,
    NavbarComponent,
    FooterComponent,
    ButtonsComponent,
    FaqComponent,
  ],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent {
  title = 'loan-predictor';
}
